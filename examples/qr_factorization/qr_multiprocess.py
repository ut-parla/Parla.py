# https://arxiv.org/pdf/1301.1071.pdf "Direct TSQR"

import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4

import sys
import getopt
import numpy as np
import concurrent.futures
import time

ROWS = 240000 # Must be >> COLS
COLS = 1000
BLOCK_SIZE = 3000 # Must be >= COLS
MAX_WORKERS = 6

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

def tsqr_blocked_multi(A, block_size, workers):
    if COLS > block_size:
        print('Block size must be greater than or equal to the number of columns in the input matrix', file=sys.stderr)
        exit(1)
    A_blocked, nblocks = make_blocked(A, block_size)
    Q1 = []
    R1 = []

    with concurrent.futures.ProcessPoolExecutor(max_workers = workers) as executor:
        # Parallel step 1
        #print('ENTERING FIRST PARALLEL SECTION')
        # Each thread gets a block from A_blocked to run numpy's build-in qr factorization on
        block_results = executor.map(np.linalg.qr, A_blocked)
        for result in block_results:
            Q1.append(result[0])
            R1.append(result[1])
        #print('FINISHED FIRST PARALLEL SECTION')

        # Sequential bottleneck
        R1 = unblock(R1)
        Q2, R = np.linalg.qr(R1) # R here is the final R result

        # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
        # Q1: block_count = A.nrows / block_size. ncols = A.ncols.
        # Q2: nrows = (A.nrows * A.ncols / block_size). Need block_count = A.nrows / block_size, nrows = A.ncols
        Q2 = make_blocked(Q2, A.shape[1])[0]

        # Parallel step 2
        #print('ENTERING SECOND PARALLEL SECTION')
        # Each thread performs a matrix multiplication call
        Q = executor.map(np.matmul, Q1, Q2)
        Q = list(Q) # Convert from a generator to a list
        #print('FINISHED SECOND PARALLEL SECTION')

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

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], 'r:c:b:w:t:')
    for opt, arg in opts:
        if opt == '-r':
            ROWS = int(arg)
        elif opt == '-c':
            COLS = int(arg)
        elif opt == '-b':
            BLOCK_SIZE = int(arg)
        elif opt == '-w':
            MAX_WORKERS = int(arg)

    print('ROWS=', ROWS, ' COLS=', COLS, ' BLOCK_SIZE=', BLOCK_SIZE, ' MAX_WORKERS=', MAX_WORKERS, sep='')
    
    for i in range(6):
        # Original matrix
        A = np.random.rand(ROWS, COLS)
        
        # Multiprocess blocked version
        start = time.time()
        Q, R = tsqr_blocked_multi(A, BLOCK_SIZE, MAX_WORKERS)
        end = time.time()
        print(f'{end - start}')
        print(check_result(A, Q, R))
