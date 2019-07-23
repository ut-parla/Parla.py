import logging
from queue import SimpleQueue
from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing.pool import ThreadPool
import threading
import time

logger = logging.getLogger(__name__)

__all__ = []

try:
    import cupy

    def get_devices():
        # Hack to get around the fact that cupy doesn't expose
        # any version of cudaGetDeviceCount.
        # 0 device is the CPU
        devices = [0]
        device_id = 0
        while True:
            next_device = cupy.cuda.Device(device_id)
            try:
                next_device.compute_capability
            except cupy.cuda.runtime.CUDARuntimeError:
                break
            device_id += 1
            devices.append(device_id)
        return devices
except ImportError:
    def get_devices():
        return [0]

# TODO: Something more intelligent here.
class HardwareTopology:
    def __init__(self):
        self.devices = get_devices()

    def num_management_threads(self):
        return len(self.devices)

topology = HardwareTopology()

thread_contexts = threading.local()

known_device_types = ["cpu", "gpu"]

def get_device_type(thread_id):
    return "gpu" if thread_id > 0 else "cpu"

class PerThreadContext():
    def __init__(self, thread_id, scheduler):
        self.thread_id = thread_id
        self.device_type = get_device_type(thread_id)
        self.scheduler = scheduler
    def __enter__(self):
        thread_contexts.context = self
    def __exit__(self):
        delattr(thread_contexts, "context")

class Scheduler:
    def __init__(self):
        self.mutex = threading.Lock()
        self.main_queue = SimpleQueue()
        self.local_queues = [SimpleQueue() for d in devices]
        self.device_queues = {device_name : SimpleQueue() for device_name in known_device_types}
        self.active = False
    def __enter__(self):
        self.active = True
    def __exit__(self):
        self.active = False
    def get(self):
        with self.mutex:
            thread_id = thread_contexts.context.thread_id
            local_queue = self.local_queues[thread_id]
            if not local_queue.empty():
                return local_queue.get()
            device_type = thread_contexts.context.device_type
            device_type_queue = device_queues[device_type]
            if not device_type_queue.empty():
                return device_type_queue.get()
            if not self.main_queue.empty():
                return self.main_queue.get()
            return None
    def put(self, task, queue_identifier=None):
        """
        Register a task as ready.
        For the time being, queue identifier can be either a thread id,
        in which case the created work item will be pushed onto
        the queue for that thread,
        a device type name, i.e. "cpu" or "gpu"
        in which case the created work item will be pushed onto
        the queue for that device type,
        or None, in which case the work item is pushed onto the main queue.
        """
        with self.mutex:
            if queue_identifer in known_device_types:
                self.device_queues[queue_identifier].put(task)
            elif queue_identifier is None:
                self.main_queue.put(task)
            else:
                self.local_queues[queue_identifier].put(task)

devices = get_devices()
main_queue = SimpleQueue()
local_queues = [SimpleQueue() for d in devices]
pool_running = False
tasks_in_progress = 0
device_indices = threading.local()
# Main thread is the CPU.
device_indices.index = 0
mutex = threading.Lock()
raised_exception = None

def get_device():
    return device_indices.index

def local_func(device_index):
    device_indices.index = device_index
    global tasks_in_progress
    global raised_exception
    local_queue = local_queues[device_index]
    # While there is any work left, do it.
    while True:
        with mutex:
            # Apparently the Queue class has it's own internal lock aside from the GIL,
            # so things can deadlock. I just went ahead and used a coarse-grained mutex
            # to guard all the scheduling logic and avoid any issues that could arise
            # from the extra queue-local lock.
            if (local_queue.empty() and main_queue.empty() and not tasks_in_progress) or raised_exception is not None:
                logger.debug("Exiting %d with (%r, %r, %r)", device_index, local_queue.empty(), tasks_in_progress, raised_exception)
                break
            if not local_queue.empty():
                work_item = local_queue.get()
            elif not main_queue.empty():
                work_item = main_queue.get()
            else:
                # TODO: intelligent backoff here?
                continue
            tasks_in_progress += 1
        # TODO: unpack args and kwargs instead of just passing a single argument.
        try:
            work_item.run()
        except Exception as exc:
            with mutex:
                if raised_exception is None:
                    raised_exception = exc
        finally:
            with mutex:
                tasks_in_progress -= 1


class Task:
    def __init__(self, func, inputs, dependencies, queue_index):
        self.func = func
        self.inputs = inputs
        self.remaining_dependencies = len(dependencies)
        self.dependees = []
        self.completed = False
        self.queue_index = queue_index
        with mutex:
            for dep in dependencies:
                if dep.completed:
                    self.remaining_dependencies -= 1
                else:
                    dep.dependees.append(self)
            if not self.remaining_dependencies:
                self.enqueue()
        # logger.info("%r", self)

    def enqueue(self):
        # Requires mutex governing queues to be held.
        receiving_queue = main_queue if self.queue_index is None else local_queues[self.queue_index]
        receiving_queue.put(self)

    def run(self):
        self.func(*self.inputs)
        with mutex:
            self.completed = True
            for dependee in self.dependees:
                dependee.remaining_dependencies -= 1
                if not dependee.remaining_dependencies:
                    dependee.enqueue()

    def __repr__(self):
        return "{func}{inputs}<{remaining_dependencies}, {completed}, {queue_index}>".format(**self.__dict__)

# Lazily starting the thread pool like this still requires the code
# to be organized so that there's a single "generation" task
# even though separate functions aren't necessary anymore.
# Maybe launching/stopping the thread pool would be better as a
# context manager.

# AMP: I think we just need to chuck the threadpool, so that we
#  can move to a customized threading system which can lazily
#  initialize and then prevent interpreter exit by having
#  non-daemon threads. The pool can shutdown based on inactivity
#  if we want, but it will use very few resources because we can
#  have "new work" condition that the threads block on and run_task
#  asserts if it is called from a non-pool thread.

def run_task(func, inputs, dependencies, queue_index = None):
    global pool_running
    global raised_exception
    if pool_running:
        return Task(func, inputs, dependencies, queue_index)
    else:
        pool_running = True
        root_task = Task(func, inputs, dependencies, queue_index)
        with ThreadPoolExecutor(len(devices)) as pool:
            local_loops = pool.map(local_func, devices)
        if raised_exception is not None:
            exc = raised_exception
            raised_exception = None
            global tasks_in_progress
            tasks_in_progress = 0
            global main_queue
            main_queue = SimpleQueue()
            pool_running = False
            global local_queues
            for i in range(len(local_queues)):
                if not local_queues[i].empty():
                    local_queues[i] = SimpleQueue()
            raise exc
        # Reset main thread to use CPU.
        device_indices.index = 0
        pool_running = False

