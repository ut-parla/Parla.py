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

class PerThreadContext:
    def __init__(self, thread_id, scheduler):
        self.thread_id = thread_id
        self.device_type = get_device_type(thread_id)
        self.scheduler = scheduler
    def __enter__(self):
        thread_contexts.context = self
    def __exit__(self, exception_type, exception_value, traceback):
        delattr(thread_contexts, "context")

def get_thread_id():
    return thread_contexts.context.thread_id

def get_device_id():
    # TODO: Make this return some kind of actual device object.
    return get_thread_id()

class Scheduler:

    def __init__(self):
        self.mutex = threading.Lock()
        self.active = False
        self.counter_mutex = threading.Lock()

    def __enter__(self):
        self.active = True
        self.main_queue = SimpleQueue()
        self.local_queues = [SimpleQueue() for d in range(topology.num_management_threads())]
        self.device_queues = {device_name : SimpleQueue() for device_name in known_device_types}
        self.tasks_in_progress = 0

    def __exit__(self, exception_type, exception_value, traceback):
        self.active = False
        del self.main_queue
        del self.local_queues
        del self.device_queues
        del self.tasks_in_progress

    def get(self):
        with self.mutex:
            assert self.active
            thread_id = get_thread_id()
            local_queue = self.local_queues[thread_id]
            if not local_queue.empty():
                return local_queue.get()
            device_type = get_device_type(get_device_id())
            device_type_queue = self.device_queues[device_type]
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
            assert self.active
            if queue_identifier in known_device_types:
                self.device_queues[queue_identifier].put(task)
            elif queue_identifier is None:
                self.main_queue.put(task)
            else:
                self.local_queues[queue_identifier].put(task)

    def run_next(self):
        next_task = self.get()
        if next_task is None:
            return False
        with self.counter_mutex:
            self.tasks_in_progress += 1
        try:
            next_task.run()
        finally:
            with self.counter_mutex:
                self.tasks_in_progress -= 1
        return True

    def finished(self):
        with self.mutex:
            assert self.active
            with self.counter_mutex:
                if self.tasks_in_progress:
                    return False
            for queue in self.local_queues:
                if not queue.empty():
                    return False
            for device_type, queue in self.device_queues.items():
                if not queue.empty():
                    return False
            if not self.main_queue.empty():
                return False
        logger.debug("Exiting %d with (%r, %d, %r)", map(len, self.local_queues), len(self.main_queue),
                     self.tasks_in_progress)
        return True

# TODO: Do we want an interface that lets the scheduler be specified at
# runtime instead of just having it be here?
scheduler = Scheduler()
# Need a lock to ensure atomicity of appending/removing from raised_exceptions.
# Could do this with atomics in a lower level language.
exception_log_mutex = threading.Lock()
raised_exceptions = []
pool_running = False

# TODO: exception handling should just be built into
# (or as a wrapper around) whatever we are using as a thread pool.
def local_func(thread_id):
    with PerThreadContext(thread_id, scheduler):
        global raised_exceptions
        while not scheduler.finished():
            with exception_log_mutex:
                if raised_exceptions:
                    logger.debug("Exiting with exceptions: {}".format(raised_exceptions))
                    break
            try:
                did_work = scheduler.run_next()
            except Exception as exc:
                with exception_log_mutex:
                    raised_exceptions.append(exc)

# Note: tasks can be implemented as lock free, however,
# atomics aren't really a thing in Python, so instead
# make each task have its own lock to mimic atomic-like
# counters for dependency tracking.

class Task:
    def __init__(self, func, inputs, dependencies, queue_identifier):
        self.func = func
        self.inputs = inputs
        self.remaining_dependencies = len(dependencies)
        self.dependees = []
        self.completed = False
        self.queue_identifier = queue_identifier
        self.mutex = threading.Lock()
        with self.mutex:
            for dep in dependencies:
                if dep.completed:
                    self.remaining_dependencies -= 1
                else:
                    dep.dependees.append(self)
            if not self.remaining_dependencies:
                self.enqueue()

    def enqueue(self):
        scheduler.put(self)

    def run(self):
        self.func(*self.inputs)
        with self.mutex:
            self.completed = True
            for dependee in self.dependees:
                dependee.remaining_dependencies -= 1
                if not dependee.remaining_dependencies:
                    dependee.enqueue()

    def __repr__(self):
        return "{func}{inputs}<{remaining_dependencies}, {completed}, {queue_identifier}>".format(**self.__dict__)

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

def run_task(func, inputs, dependencies, queue_identifier = None):
    global pool_running
    if pool_running:
        return Task(func, inputs, dependencies, queue_identifier)
    else:
        pool_running = True
        with scheduler, ThreadPoolExecutor(topology.num_management_threads()) as pool:
            root_task = Task(func, inputs, dependencies, queue_identifier)
            local_loops = pool.map(local_func, range(topology.num_management_threads()))
        pool_running = False
        global raised_exceptions
        with exception_log_mutex:
            if raised_exceptions:
                # TODO: Handle multiple exception case better
                exc = raised_exceptions[0]
                raised_exceptions = []
                raise exc
        pool_running = False
        return root_task
