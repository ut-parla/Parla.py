import numpy as np
from time import perf_counter as time
import mkl
import concurrent.futures

MATRIX_COUNTS = [2**i for i in range(7, 12)]
MATRIX_SIZES  = MATRIX_COUNTS[::-1]
THREAD_COUNTS = [24, 12, 8, 6, 4]

print(MATRIX_COUNTS)
print(MATRIX_SIZES)

mats = [[None for i in range(count)] for count in MATRIX_COUNTS]

for i, s in enumerate(MATRIX_SIZES):
    for j in range(MATRIX_COUNTS[i]):
        mats[i][j] = np.random.rand(s, s)

def worker(mats, i, j):
    mats[i][j] = np.matmul(mats[i][j], mats[i][j])
    
start = time()
for i, s in enumerate(MATRIX_SIZES):
    mkl.set_num_threads(THREAD_COUNTS[i])
    with concurrent.futures.ThreadPoolExecutor(max_workers=24 / THREAD_COUNTS[i]) as executor:
        for j in range(MATRIX_COUNTS[i]):
            executor.submit(worker, mats, i, j)

total = time() - start

print(total)
