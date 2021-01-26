import numpy as np
import time

ROWS = 240000 # Must be >> COLS
COLS = 1000
BLOCK_SIZE = 3000

def check_result(A, Q, R):
    # Check product
    is_correct_prod = np.allclose(np.matmul(Q, R), A)
    
    # Check orthonormal
    Q_check = np.matmul(Q.transpose(), Q)
    is_ortho_Q = np.allclose(Q_check, np.identity(COLS))
    
    # Check upper
    is_upper_R = np.allclose(R, np.triu(R))

    return is_correct_prod and is_ortho_Q and is_upper_R

for i in range(6):
    # Original matrix
    A = np.random.rand(ROWS, COLS)
    
    # Numpy version
    start = time.time()
    Q, R = np.linalg.qr(A)
    end = time.time()
    print(end - start)
    print(check_result(A, Q, R))
