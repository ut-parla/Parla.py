import argparse
import numpy as np
import dask, dask.array as da
from time import time

def check_result(A, Q, R):
    # Check product
    is_correct_prod = np.allclose(np.matmul(Q, R), A)
    
    # Check orthonormal
    Q_check = np.matmul(Q.transpose(), Q)
    is_ortho_Q = np.allclose(Q_check, np.identity(NCOLS))
    
    # Check upper
    is_upper_R = np.allclose(R, np.triu(R))

    return is_correct_prod and is_ortho_Q and is_upper_R

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rows", help="Number of rows for input matrix; must be >> cols", type=int, default=5000)
    parser.add_argument("-c", "--cols", help="Number of columns for input matrix", type=int, default=100)
    parser.add_argument("-b", "--block_size", help="Block size to break up input matrix; must be >= cols", type=int, default=500)
    parser.add_argument("-i", "--iterations", help="Number of iterations to run experiment. If > 1, first is ignored as warmup.", type=int, default=1)
    parser.add_argument("-K", "--check_result", help="Checks final result on CPU", action="store_true")

    args = parser.parse_args()

    # Set global config variables
    NROWS = args.rows
    NCOLS = args.cols
    BLOCK_SIZE = args.block_size
    ITERS = args.iterations
    CHECK_RESULT = args.check_result

    print('%**********************************************************************************************%\n')
    print('Config: rows=', NROWS, ' cols=', NCOLS, ' iterations=', ITERS, ' check_result=', CHECK_RESULT, sep='', end='\n\n')
    times = [None] * ITERS

    # Dask version
    for i in range(ITERS):
        # Original matrix
        A = da.random.random((NROWS, NCOLS), chunks=(BLOCK_SIZE, NCOLS))
        A.persist()

        start = time()
        Q, R = da.linalg.qr(A)
        dask.compute(Q, R)
        end = time()
        times[i] = end - start

        if CHECK_RESULT:
            if check_result(A, Q, R):
                print("\nCorrect result!\n")
            else:
                print("%***** ERROR: Incorrect final result!!! *****%")

    if ITERS > 1:
        times = times[1:]

    print(times)
    print("Average:", np.average(times))
    print("Std dev:", np.std(times))
