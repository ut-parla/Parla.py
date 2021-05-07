import os
os.environ["OMP_NUM_THREADS"] = "24" # This is the default on my machine (Zemaitis)
import numpy as np
from time import perf_counter as time

NMATRICES = 8
MATRIX_SIZES  = [512, 4096]

small_mats = [np.random.rand(MATRIX_SIZES[0], MATRIX_SIZES[0]) for i in range(NMATRICES)]
large_mat = np.random.rand(MATRIX_SIZES[1], MATRIX_SIZES[1])

print("Starting computation")
start = time()
large_mat = np.matmul(large_mat, large_mat)
for i in range(NMATRICES):
    small_mats[i] = np.matmul(small_mats[i], small_mats[i])
total = time() - start

print(total)
