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

def enqueue_ready_task(task):
    queue_index = task.queue_index
    receiving_queue = main_queue if queue_index is None else local_queues[queue_index]
    receiving_queue.put(task)

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
        work_item.func(work_item.inputs)
        work_item.completed = True
        for dependee in work_item.dependees:
            dependee.remaining_dependencies -= 1
            if not dependee.remaining_dependencies:
                enqueue_ready_task(dependee)
        tasks_in_progress -= 1

class Task:
    pass

def create_task_inside_pool(func, inputs, dependencies, queue_index):
    created_item = Task()
    created_item.func = func
    created_item.inputs = inputs
    created_item.remaining_dependencies = len(dependencies)
    created_item.dependees = []
    created_item.completed = False
    created_item.queue_index = queue_index
    for dep in dependencies:
        if dep.completed:
            created_item.remaining_dependencies -= 1
        else:
            dep.dependees.append(created_item)
    if created_item.remaining_dependencies:
        return created_item
    enqueue_ready_task(created_item)
    return created_item

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
        return create_task_inside_pool(func, inputs, dependencies, queue_index)
    else:
        pool_running = True
        create_task_inside_pool(func, inputs, dependencies, queue_index)
        with ThreadPoolExecutor(len(devices)) as pool:
            pool.map(local_func, devices)
        pool_running = False

