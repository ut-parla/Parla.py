# https://arxiv.org/pdf/1301.1071.pdf "Direct TSQR"

import sys
import numpy as np
import time

ROWS = 240000 # Must be >> COLS
COLS = 1000
BLOCK_SIZE = 3000

# Accepts a matrix and returns a list of its blocks
# block_size rows are grouped together
def make_blocked(A, block_size):
    nrows = A.shape[0]
    nblocks = (nrows + block_size - 1) // block_size # ceiling division

    block_list = []

    for i in range(0, nblocks):
        lower = i * block_size; # first row in block, inclusive
        upper = (i + 1) * block_size # last row in block, exclusive
        if upper > nrows:
            upper = nrows

        block_list.append(A[lower:upper])

    return block_list, nblocks

# Get back to original matrix form
def unblock(A):
    return np.concatenate(A)

def tsqr_blocked_single(A, block_size):
    if COLS > block_size:
        print('Block size must be greater than or equal to the number of columns in the input matrix', file=sys.stderr)
        exit(1)
    A_blocked, nblocks = make_blocked(A, block_size)
    Q1 = []
    R1 = []
    for block in A_blocked:
        # Use numpy's built in qr for the base factorization
        block_Q, block_R = np.linalg.qr(block)
        Q1.append(block_Q)
        R1.append(block_R)

    R1 = unblock(R1)

    # R here is the final R result
    Q2, R = np.linalg.qr(R1)

    # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
    # Q2 is currently an (ncols * nblocks) x ncols matrix. Need nblocks of ncols rows each
    Q2 = make_blocked(Q2, A.shape[1])[0]

    Q = [np.matmul(Q1[i], Q2[i]) for i in range(nblocks)]
    Q = unblock(Q)
    
    return Q, R

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
    
    # Blocked version without nested parallelism
    start = time.time()
    Q, R = tsqr_blocked_single(A, BLOCK_SIZE)
    end = time.time()
    print(end - start)
    print(check_result(A, Q, R))
