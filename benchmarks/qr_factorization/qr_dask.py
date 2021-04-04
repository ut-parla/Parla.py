import dask, dask.array as da
import numpy as np
import cupy as cp
import time

ROWS = 2000000 # Must be >> COLS
COLS = 1000
BLOCK_SIZE = 400000

if __name__ == "__main__":
    # Create random matrix. Use np for CPU or cp for GPU
    rs = dask.array.random.RandomState(RandomState=cp.random.RandomState)
    A = rs.random((ROWS, COLS), chunks=(BLOCK_SIZE, COLS))
    A.persist()

    start = time.time()
    Q, R = da.linalg.qr(A)
    Q.compute()
    end = time.time()
    print(end - start)
