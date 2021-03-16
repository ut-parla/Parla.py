import time

import numpy as np
from sleep.core import sleep, bsleep

from parla import Parla
from parla.cpu import cpu
from parla.tasks import spawn
from numba import jit, njit

tt = 1000
rep = 1

def branch(depth=0, threshold=5):
    a = 0
    
    t = time.time()
    a = 0
    for k in range(rep): 
        bsleep(tt//rep)
        a+=1
    et = time.time()
    print(depth, " :: ", et-t, flush=True)

    if depth+1 > threshold:
        return

    @spawn(placement=cpu)
    def lower_block_task():
        branch(depth=depth+1, threshold=threshold)
    @spawn(placement=cpu)
    def upper_block_task():
        branch(depth=depth+1, threshold=threshold)


if __name__ == '__main__':
    max_depth = 2
    nodes = 2**max_depth - 1
    sequential_time = nodes*tt
    
    start = time.perf_counter()
    with Parla():
        branch(threshold=max_depth)
    end = time.perf_counter()
    print("Elapsed: ", end - start, "seconds")
    print("Sequential Time: ", sequential_time)
