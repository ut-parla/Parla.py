from parla.multiload import multiload_contexts as VECs

#import numpy as np # Do this later so we can set the threadpool size
from time import perf_counter as time
import concurrent.futures

show_backtrace()

NMATRICES = 10
MATRIX_SIZES  = [2, 4]
#NMATRICES = 10
#MATRIX_SIZES  = [128, 2048]
SMALL_THREAD_COUNT = 8
LARGE_THREAD_COUNT = 16

small_VEC = VECs[0]
small_VEC.setenv('OMP_NUM_THREADS', str(SMALL_THREAD_COUNT))
with small_VEC:
    import numpy as np
    small_mats = [np.random.rand(MATRIX_SIZES[0], MATRIX_SIZES[0]) for i in range(NMATRICES)]

large_VEC = VECs[1]
large_VEC.setenv('OMP_NUM_THREADS', str(LARGE_THREAD_COUNT))
with large_VEC:
    import numpy as np
    large_mats = [np.random.rand(MATRIX_SIZES[1], MATRIX_SIZES[1]) for i in range(NMATRICES)]

def worker(mats, i, VEC):
    print(mats)
    print(i)
    print(VEC)
    with VEC:
        print('start')
        A = mats[i]
        print('read complete')
        B = np.matmul(A, A)
        print('matmul complete')
        print(B)
        return B
    
print("Starting computation")
start = time()
for i in range(NMATRICES):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = executor.submit(worker)
        large_mats[i] = executor.submit(worker, large_mats, i, large_VEC)
        small_mats[i] = executor.submit(worker, small_mats, i, small_VEC)
    print(i, "done")
total = time() - start

print(total)

with small_VEC:
    print(small_mats[i].result())
