import numpy as np
import dask, dask.array as da
import time

ROWS = 240000 # Must be >> COLS
COLS = 1000
BLOCK_SIZE = 3000

for i in range(6):
    # Original matrix
    A = da.random.random((ROWS, COLS), chunks=(BLOCK_SIZE, COLS))
    A.compute()

    # Dask version
    start = time.time()
    Q, R = da.linalg.qr(A)
    Q.compute()
    end = time.time()
    print(end - start)
