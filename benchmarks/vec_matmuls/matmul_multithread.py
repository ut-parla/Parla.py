import os
os.environ["OMP_NUM_THREADS"] = "24" # TODO: Play with this for best performance
import numpy as np
from time import perf_counter as time
import threading

NMATRICES = 8
MATRIX_SIZES  = [512, 4096]

small_mats = [np.random.rand(MATRIX_SIZES[0], MATRIX_SIZES[0]) for i in range(NMATRICES)]
large_mats = []
large_mats.append(np.random.rand(MATRIX_SIZES[1], MATRIX_SIZES[1]))

def small_worker(i):
    small_mats[i] = np.matmul(small_mats[i], small_mats[i])

def large_worker(large_mats):
    large_mats[0] = np.matmul(large_mats[0], large_mats[0])
    
small_threads = [None] * NMATRICES

print("Starting computation")
start = time()
t_large = threading.Thread(target=large_worker, args=(large_mats,))
t_large.start()
for i in range(NMATRICES):
    small_threads[i] = threading.Thread(target=small_worker, args=(i,))
    small_threads[i].start()
for i in range(NMATRICES):
    small_threads[i].join()
t_large.join()
total = time() - start

print(total)
