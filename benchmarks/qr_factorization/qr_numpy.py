import os
os.environ["OMP_NUM_THREADS"] = "24" # This is the default on my machine (Zemaitis)
import argparse
import numpy as np
import scipy.linalg
from time import perf_counter as time

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
    parser.add_argument("-i", "--iterations", help="Number of iterations to run experiment.", type=int, default=1)
    parser.add_argument("-w", "--warmup", help="Number of warmup runs to perform before iterations.", type=int, default=0)
    parser.add_argument("-K", "--check_result", help="Checks final result on CPU", action="store_true")

    args = parser.parse_args()

    # Set global config variables
    NROWS = args.rows
    NCOLS = args.cols
    ITERS = args.iterations
    WARMUP = args.warmup
    CHECK_RESULT = args.check_result

    print('%**********************************************************************************************%\n')
    print('Config: rows=', NROWS, ' cols=', NCOLS, ' iterations=', ITERS, ' warmup=', WARMUP, ' check_result=', CHECK_RESULT, sep='', end='\n\n')

    for i in range(WARMUP + ITERS):
        # Original matrix
        A = np.random.rand(NROWS, NCOLS)

        start = time()
        Q, R = scipy.linalg.qr(A, mode='economic')
        end = time()
        if (i >= WARMUP):
            print(end - start)

        if CHECK_RESULT:
            if check_result(A, Q, R):
                print("\nCorrect result!\n")
            else:
                print("%***** ERROR: Incorrect final result!!! *****%")
