import os
os.environ["OMP_NUM_THREADS"] = "24" # This is the default on my machine (Zemaitis)
import numpy as np
from time import perf_counter as time

NMATRICES = 100
MATRIX_SIZES  = [128, 2048]

mats = [[np.random.rand(s, s) for i in range(NMATRICES)] for s in MATRIX_SIZES]

print("Starting computation")
start = time()
for i in range(NMATRICES):
    mats[0][i] = np.matmul(mats[0][i], mats[0][i])
    mats[1][i] = np.matmul(mats[1][i], mats[1][i])
total = time() - start

print(total)
