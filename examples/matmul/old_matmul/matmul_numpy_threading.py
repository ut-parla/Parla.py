
import mkl
#Set the number of MKL threads
mkl.set_num_threads(4)


#This feels really bad but its the easy way to change the demo
import numpy as cp
#import cupy as cp
import time 
import numpy as np

import sys
from parla import Parla, get_all_devices
from parla.array import copy, clone_here
from parla.cpu import cpu
from parla.cuda import gpu
from parla.function_decorators import specialized
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import spawn, TaskSpace, CompletedTaskSpace

import threading 


#Set the number of threads that will be spawned
ngpus = 2

# set up two n x n arrays to multiply together.
# n is chosen so that all three can be
# stored within the memory of a single GPU
# so that strong scaling numbers make sense.
n = 20000
np.random.seed(0)
a_cpu = np.random.rand(n, n).astype(np.float32)
b_cpu = np.random.rand(n, n).astype(np.float32)
# Partition the two arrays and set up the
# partitioned array where the result will be stored.
# This could also be done using a parla mapper object.
a_part = []
b_part = []
c_part = []
block_size = n // ngpus + 1
for i in range(ngpus):
    a_part.append(cp.array(a_cpu[i * block_size : (i + 1) * block_size]))
    b_part.append(cp.array(b_cpu[i * block_size : (i + 1) * block_size]))
    c_part.append(cp.empty_like(b_part[-1]))

print("Nthreads", ngpus)

start = time.perf_counter()

def worker(i, a_block, c_part):
    #start_t = time.time()
    c_block = c_part[:, i * block_size : (i + 1) * block_size]
    #print("Sizes:", a_block.shape, c_block.shape, flush=True)
    start_t = time.time()
    L = a_block @ a_block.T
    end_t = time.time()
    print(end_t - start_t, flush=True)
    return 

start = time.perf_counter()

thread_list = []
for i in range(ngpus):
    t = threading.Thread(target=worker, args=(i, a_part[i], c_part[i]))
    thread_list.append(t)

start = time.perf_counter()
for t in thread_list:
    t.start()

#thread_list[0].join()
for t in thread_list:
    t.join()

end = time.perf_counter()

print("Total Time:", end - start, flush=True)
