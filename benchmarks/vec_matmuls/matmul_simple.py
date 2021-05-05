import numpy as np
from time import perf_counter as time

MATRIX_COUNTS = [2**i for i in range(7, 12)]
MATRIX_SIZES  = MATRIX_COUNTS[::-1]

print(MATRIX_COUNTS)
print(MATRIX_SIZES)

mats = [[None for i in range(count)] for count in MATRIX_COUNTS]

for i, s in enumerate(MATRIX_SIZES):
    for j in range(MATRIX_COUNTS[i]):
        mats[i][j] = np.random.rand(s, s)

start = time()
for i, s in enumerate(MATRIX_SIZES):
    for j in range(MATRIX_COUNTS[i]):
        mats[i][j] = np.matmul(mats[i][j], mats[i][j])
total = time() - start

print(total)
