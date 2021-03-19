import numpy as np
import cupy as cp
import time

ROWS = 500000 # Must be >> COLS
COLS = 1000

def check_result(A, Q, R):
    # Check product
    is_correct_prod = np.allclose(np.matmul(Q, R), A)
    
    # Check orthonormal
    Q_check = np.matmul(Q.transpose(), Q)
    is_ortho_Q = np.allclose(Q_check, np.identity(COLS))
    
    # Check upper
    is_upper_R = np.allclose(R, np.triu(R))

    return is_correct_prod and is_ortho_Q and is_upper_R

if __name__ == "__main__":
    times_H2D = [None] * 6
    times_ker = [None] * 6
    times_D2H = [None] * 6

    for i in range(6):
        # Original matrix
        A = np.random.rand(ROWS, COLS)
    
        # Cupy version
        start = time.time()
        A_GPU = cp.array(A)
        end = time.time()
        times_H2D[i] = end - start
    
        start = time.time()
        Q, R = cp.linalg.qr(A_GPU)
        end = time.time()
        times_ker[i] = end - start

        start = time.time()
        Q = cp.asnumpy(Q)
        R = cp.asnumpy(R)
        end = time.time()
        times_D2H[i] = end - start

        #print(check_result(A, Q, R))

    times_H2D = times_H2D[1:]
    times_ker = times_ker[1:]
    times_D2H = times_D2H[1:]

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

