from parla.multiload import multiload_contexts as VECs
#import parla.vec_backtrace

#import numpy as np # Do this later so we can set the threadpool size
from time import perf_counter as time
import threading

NMATRICES = 8
MATRIX_SIZES  = [512, 4096]
SMALL_THREAD_COUNT = 1
LARGE_THREAD_COUNT = 23

small_VEC = VECs[0]
small_VEC.setenv('OMP_NUM_THREADS', str(SMALL_THREAD_COUNT))
with small_VEC:
    import numpy as np
    import scipy.linalg
    small_mats = [np.random.rand(MATRIX_SIZES[0], MATRIX_SIZES[0]) for i in range(NMATRICES)]

large_VEC = VECs[1]
large_VEC.setenv('OMP_NUM_THREADS', str(LARGE_THREAD_COUNT))
with large_VEC:
    import numpy as np
    import scipy.linalg
    large_mats = []
    large_mats.append(np.random.rand(MATRIX_SIZES[1], MATRIX_SIZES[1]))

def small_worker(i):
    with small_VEC:
        small_mats[i] = np.matmul(small_mats[i], small_mats[i])

def large_worker(large_mats):
    with large_VEC:
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
print(total, flush=True)
