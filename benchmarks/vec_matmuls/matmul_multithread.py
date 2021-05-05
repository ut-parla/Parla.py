import os
os.environ["OMP_NUM_THREADS"] = "24" # TODO: Play with this for best performance
import numpy as np
from time import perf_counter as time
import threading

NMATRICES = 100
MATRIX_SIZES  = [128, 2048]

mats = [[np.random.rand(s, s) for i in range(NMATRICES)] for s in MATRIX_SIZES]

def worker(mats, i, j):
    mats[i][j] = np.matmul(mats[i][j], mats[i][j])
    
print("Starting computation")
start = time()
for i in range(NMATRICES):
    t_small = threading.Thread(target=worker, args=(mats, 0, i))
    t_large = threading.Thread(target=worker, args=(mats, 1, i))
    t_large.start()
    t_small.start()
    t_small.join()
    t_large.join()
total = time() - start

print(total)
