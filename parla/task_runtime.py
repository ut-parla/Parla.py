from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
import cupy
import threading
import time

# Note: The code here relies on the semantics of the GIL to ensure thread safety in various places.

def get_devices():
    # Hack to get around the fact that cupy doesn't expose
    # any version of cudaGetDeviceCount.
    # 0 device is the CPU
    devices = [0]
    device_id = 1
    while True:
        next_device = cupy.cuda.Device(device_id-1)
        try:
            next_device.compute_capability
        except cupy.cuda.runtime.CUDARuntimeError:
            break
        device_id += 1
        devices.append(device_id)
    return devices

devices = get_devices()
main_queue = Queue()
local_queues = [Queue() for d in devices]
pool_running = False
tasks_in_progress = 0

def local_func(device_index):
    global tasks_in_progress
    local_queue = local_queues[device_index]
    # While there is any work left, do it.
    while local_queue or main_queue or tasks_in_progress:
        if local_queue:
            work_item = local_queue.popleft()
        elif main_queue:
            work_item = main_queue.popleft()
        else:
            # TODO: intelligent backoff here
            time.sleep(.005)
            continue
        tasks_in_progress += 1
        # TODO: unpack args and kwargs instead of just passing a single argument.
        work_item.run()
        tasks_in_progress -= 1

class Task:

    def __init__(self, func, inputs, dependencies, queue_index):
        self.func = func
        self.inputs = inputs
        self.remaining_dependencies = len(dependencies)
        self.dependees = []
        self.completed = False
        self.queue_index = queue_index
        for dep in dependencies:
            if dep.completed:
                self.remaining_dependencies -= 1
            else:
                dep.dependees.append(self)
        if not self.remaining_dependencies:
            self.enqueue()

    def enqueue(self):
        receiving_queue = main_queue if self.queue_index is None else local_queues[queue_index]
        receiving_queue.put(self)

    def run(self):
        self.func(*self.inputs)
        self.completed = True
        for dependee in work_item.dependees:
            dependee.remaining_dependencies -= 1
            if not dependee.remaining_dependencies:
                dependee.enqueue()

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
    if pool_running:
        return Task(func, inputs, dependencies, queue_index)
    else:
        pool_running = True
        root_task = Task(func, inputs, dependencies, queue_index)
        with ThreadPoolExecutor(len(devices)) as pool:
            pool.map(local_func, devices)
        pool_running = False

