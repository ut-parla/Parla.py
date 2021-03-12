import numpy as np
import time

ROWS = 2000000 # Must be >> COLS
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
    # Original matrix
    A = np.random.rand(ROWS, COLS)
    
    # Numpy version
    for i in range(6):
      start = time.time()
      Q, R = np.linalg.qr(A)
      end = time.time()
      print(end - start)
      #print(check_result(A, Q, R))
