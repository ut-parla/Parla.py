import collections
from concurrent.futures import ThreadPoolExecutor
import cupy
import threading
import time

def get_devices():
    # Hack to get around the fact that cupy doesn't expose
    # any version of cudaGetDeviceCount.
    # "None" device is CPU
    devices = [None]
    device_id = 0
    while True:
        next_device = cupy.cuda.Device(device_id)
        try:
            next_device.compute_capability
        except cupy.cuda.runtime.CUDARuntimeError:
            break
        device_id += 1
        devices.append(next_device)
    return devices

devices = get_devices()
pool = ThreadPoolExecutor(len(devices))
main_queue = collections.deque()
local_queues = [collections.deque() for d in devices]

def local_func(device_info):
    index, device = device_info
    local_queue = local_queue[index]
    # While there is any work left, do it.
    while local_queue or main_queue or tasks_in_progress:
        if local_queue:
            op, closure = local_queue.popleft()
        elif main_queue:
            op, closure = main_queue.popleft()
        else:
            # TODO: intelligent backoff here
            time.sleep(5)
            continue
        tasks_in_progress += 1
        op(closure)
        tasks_in_progress -= 1

pool.map(local_func, devices)

def run_generation_task(func, inputs):
    # Global counter used for termination detection
    tasks_in_progress = 0
    # Outer loop run on the managing thread for each device
    with ThreadPoolExecutor(len(devices)) as pool:
        pool.map(local_func, enumerate(devices))


# Dummy type to allow for sanity checks in other task handling code.
Task = tuple