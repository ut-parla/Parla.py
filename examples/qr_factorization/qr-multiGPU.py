# https://arxiv.org/pdf/1301.1071.pdf "Direct TSQR"

import sys
import numpy as np
#import cupy as cp
import time

from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import *

ROWS = 240 # Must be >> COLS
COLS = 10
BLOCK_SIZE = 30

# Accepts a matrix and returns a list of its blocks and the block count
# block_size rows are grouped together
# I think this doesn't copy?
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

# Get back to original matrix form. Pass in a blocked matrix as a list.
def unblock(A):
    return np.concatenate(A)

# A: 2D numpy matrix
# block_size: Positive integer
async def tsqr_blocked(A, block_size):
    # Check for block_size > ncols
    assert A.shape[1] <= block_size, "Block size must be greater than or equal to the number of columns in the input matrix"

    # Break input into blocks (stored as list of ndarrays)
    A_blocked, nblocks = make_blocked(A, block_size)

    # Initialize empty lists to store blocks
    Q1_blocked = [None] * nblocks;
    R1_blocked = [None] * nblocks;
    #Q2_blocked = None
    Q2_blocked = 1

    # Create tasks to perform qr factorization on each block and store them in lists
    T1 = TaskSpace()
    for i in range(nblocks):
        @spawn(taskid=T1[i])
        def t1():
            Q1_blocked[i], R1_blocked[i] = np.linalg.qr(A_blocked[i])

    # Perform intermediate qr factorization on R1 to get Q2 and final R
    @spawn(dependencies=T1)
    def t2():
        R1 = unblock(R1_blocked)

        # R here is the final R result
        Q2, R = np.linalg.qr(R1)

        # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
        # Q2 is currently an (ncols * nblocks) x ncols matrix. Need nblocks of ncols rows each
        Q2_blocked = make_blocked(Q2, A.shape[1])[0]
        return Q2_blocked, R

    Q2_blocked, R= await t2

    # Create tasks to perform Q1 @ Q2 matrix multiplication by block
    Q_blocked = [None] * nblocks
    T3 = TaskSpace()
    for i in range(nblocks):
        @spawn(taskid=T3[i], dependencies=[T1[i], t2])
        def t3():
            Q_blocked[i] = np.matmul(Q1_blocked[i], Q2_blocked[i])

    await T3
    Q = unblock(Q_blocked)
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

def main():
    @spawn()
    async def test_tsqr_blocked():
        np.random.seed(0)

        # Original matrix
        A = np.random.rand(ROWS, COLS)
        
        # Blocked version without nested parallelism
        start = time.time()
        Q, R = await tsqr_blocked(A, BLOCK_SIZE)
        end = time.time()
        print(end - start)
        print(check_result(A, Q, R))
    
if __name__ == "__main__":
    with Parla():
        main()
