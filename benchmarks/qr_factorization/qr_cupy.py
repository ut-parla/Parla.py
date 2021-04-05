import argparse
import numpy as np
import cupy as cp
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
    parser.add_argument("-i", "--iterations", help="Number of iterations to run experiment. If > 1, first is ignored as warmup.", type=int, default=1)
    parser.add_argument("-K", "--check_result", help="Checks final result on CPU", action="store_true")

    args = parser.parse_args()

    # Set global config variables
    NROWS = args.rows
    NCOLS = args.cols
    ITERS = args.iterations
    CHECK_RESULT = args.check_result

    print('%**********************************************************************************************%\n')
    print('Config: rows=', NROWS, ' cols=', NCOLS, ' iterations=', ITERS, ' check_result=', CHECK_RESULT, sep='', end='\n\n')

    times_H2D = [None] * ITERS
    times_ker = [None] * ITERS
    times_D2H = [None] * ITERS

    for i in range(ITERS):
        # Original matrix
        A = np.random.rand(NROWS, NCOLS)
    
        # Cupy version
        start = time()
        A_GPU = cp.array(A)
        cp.cuda.stream.get_current_stream().synchronize()
        end = time()
        times_H2D[i] = end - start
    
        start = time()
        Q, R = cp.linalg.qr(A_GPU)
        cp.cuda.stream.get_current_stream().synchronize()
        end = time()
        times_ker[i] = end - start

        start = time()
        Q = cp.asnumpy(Q)
        R = cp.asnumpy(R)
        cp.cuda.stream.get_current_stream().synchronize()
        end = time()
        times_D2H[i] = end - start

        if CHECK_RESULT:
            if check_result(A, Q, R):
                print("\nCorrect result!\n")
            else:
                print("%***** ERROR: Incorrect final result!!! *****%")

    if ITERS > 1:
        times_H2D = times_H2D[1:]
        times_ker = times_ker[1:]
        times_D2H = times_D2H[1:]
        times_total = [times_H2D[i] + times_ker[i] + times_D2H[i] for i in range(ITERS - 1)]
    else:
        times_total = [times_H2D[i] + times_ker[i] + times_D2H[i] for i in range(ITERS)]

    print("Host to Device")
    print(times_H2D)
    print("Average:", np.average(times_H2D))
    print("Std dev:", np.std(times_H2D))
    print()

    print("Kernel")
    print(times_ker)
    print("Average:", np.average(times_ker))
    print("Std dev:", np.std(times_ker))
    print()

    print("Device to Host")
    print(times_D2H)
    print("Average:", np.average(times_D2H))
    print("Std dev:", np.std(times_D2H))

    print("Total")
    print(times_total)
    print("Average:", np.average(times_total))
    print("Std dev:", np.std(times_total))
