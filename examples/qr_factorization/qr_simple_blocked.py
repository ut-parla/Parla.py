# https://arxiv.org/pdf/1301.1071.pdf "Direct TSQR"

import os
os.environ["OMP_NUM_THREADS"] = "24" # This is the default on my machine (Zemaitis)
import sys
import argparse
import numpy as np
import scipy.linalg
from time import perf_counter as time

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

def tsqr_blocked(A):
    if NCOLS > BLOCK_SIZE:
        print('Block size must be greater than or equal to the number of columns in the input matrix', file=sys.stderr)
        exit(1)
    A_blocked, nblocks = make_blocked(A, BLOCK_SIZE)
    Q1 = []
    R1 = []

    for block in A_blocked:
        # Use numpy's built in qr for the base factorization
        block_Q, block_R = scipy.linalg.qr(block, mode='economic')
        Q1.append(block_Q)
        R1.append(block_R)

    R1 = unblock(R1)

    # R here is the final R result
    Q2, R = scipy.linalg.qr(R1, mode='economic')

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
    is_ortho_Q = np.allclose(Q_check, np.identity(NCOLS))
    
    # Check upper
    is_upper_R = np.allclose(R, np.triu(R))

    return is_correct_prod and is_ortho_Q and is_upper_R

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rows", help="Number of rows for input matrix; must be >> cols", type=int, default=5000)
    parser.add_argument("-c", "--cols", help="Number of columns for input matrix", type=int, default=100)
    parser.add_argument("-b", "--block_size", help="Block size to break up input matrix; must be >= cols", type=int, default=500)
    parser.add_argument("-i", "--iterations", help="Number of iterations to run experiment", type=int, default=1)
    parser.add_argument("-w", "--warmup", help="Number of warmup runs to perform before the experiment", type=int, default=0)
    parser.add_argument("-K", "--check_result", help="Checks final result on CPU", action="store_true")
    parser.add_argument("--csv", help="Prints stats in csv format", action="store_true")

    args = parser.parse_args()

    # Set global config variables
    NROWS = args.rows
    NCOLS = args.cols
    BLOCK_SIZE = args.block_size
    ITERS = args.iterations
    WARMUP = args.warmup
    CHECK_RESULT = args.check_result
    CSV = args.csv

    print('%**********************************************************************************************%\n')
    print('Config: rows=', NROWS, ' cols=', NCOLS, ' block_size=', BLOCK_SIZE, ' iterations=', ITERS, ' warmup=', WARMUP, \
        ' check_result=', CHECK_RESULT, ' csv=', CSV, sep='')
    
    for i in range(WARMUP + ITERS):
        # Original matrix
        np.random.seed(i)
        A = np.random.rand(NROWS, NCOLS)

        # Multithreaded blocked version with VECs
        start = time()
        Q, R = tsqr_blocked(A)
        end = time()

        if (i >= WARMUP):
            print(end - start)

        if CHECK_RESULT:
            print(check_result(A, Q, R))

    print('\n%**********************************************************************************************%\n')
