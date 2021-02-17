# https://arxiv.org/pdf/1301.1071.pdf "Direct TSQR"

import sys
import numpy as np
import cupy as cp
import time

from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.array import clone_here
from parla.function_decorators import specialized
from parla.tasks import *

ROWS = 2000000 # Must be >> COLS
COLS = 1000
BLOCK_SIZE = 400000

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

# Not used
@specialized
def qr_block(block):
    return np.linalg.qr(block)

@qr_block.variant(gpu)
def qr_block_gpu(block):
    gpu_Q, gpu_R = cp.linalg.qr(block)
    cpu_Q = cp.asnumpy(gpu_Q)
    cpu_R = cp.asnumpy(gpu_R)
    return cpu_Q, cpu_R

# Not used
@specialized
def matmul_block(block_1, block_2):
    return np.matmul(block_1, block_2)

@matmul_block.variant(gpu)
def matmul_block_gpu(block_1, block_2):
    gpu_Q = cp.matmul(block_1, block_2)
    cpu_Q = cp.asnumpy(gpu_Q)
    return cpu_Q

# A: 2D numpy matrix
# block_size: Positive integer
async def tsqr_blocked(A, block_size):
    nrows, ncols = A.shape

    # Check for block_size > ncols
    assert ncols <= block_size, "Block size must be greater than or equal to the number of columns in the input matrix"

    # Break input into blocks (stored as list of ndarrays)
    A_blocked, nblocks = make_blocked(A, block_size)

    # Initialize empty lists to store blocks
    Q1_blocked = [None] * nblocks; # Doesn't need to be contiguous, so basically just pointers
    R1 = np.empty([nblocks * ncols, ncols]) # Needs to be allocated contiguously for later concatenation
    R1_blocked = make_blocked(R1, ncols)[0] # R1 has nblocks, each of size ncols * ncols
    # Q2 is allocated and blocked in t2
    Q = np.empty([nrows, ncols]) # Needs to be allocated contiguously for later concatenation
    Q_blocked = make_blocked(Q, block_size)[0] # Q has the same block count and dimensions as A

    # Create tasks to perform qr factorization on each block and store them in lists
    T1 = TaskSpace()
    for i in range(nblocks):
        @spawn(taskid=T1[i], placement=gpu, memory=(3*A_blocked[i].nbytes))
        def t1():
            print("t1[", i, "] start", sep='')
            block_local = clone_here(A_blocked[i])
            Q1_blocked[i], R1_blocked[i] = qr_block(block_local)
            print("t1[", i, "] end", sep='')

    # Perform intermediate qr factorization on R1 to get Q2 and final R
    @spawn(dependencies=T1)
    def t2():
        print("t2 start")
        R1 = unblock(R1_blocked) # This should be zero-copy since R1_blocked was allocated contiguously

        # R here is the final R result
        Q2, R = np.linalg.qr(R1) # TODO Do this recursively or on the GPU if it's slow?

        # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
        # Q2 is currently an (ncols * nblocks) x ncols matrix. Need nblocks of ncols rows each
        Q2_blocked = make_blocked(Q2, A.shape[1])[0]
        return Q2_blocked, R

    Q2_blocked, R = await t2
    print("t2 end")

    # Create tasks to perform Q1 @ Q2 matrix multiplication by block
    T3 = TaskSpace()
    for i in range(nblocks):
        @spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=gpu, memory=(3*Q1_blocked[i].nbytes))
        def t3():
            print("t3[", i, "] start", sep='')
            Q1_block_local = clone_here(Q1_blocked[i])
            Q2_block_local = clone_here(Q2_blocked[i])
            Q_blocked[i] = matmul_block(Q1_block_local, Q2_block_local)
            print("t3[", i, "] end", sep='')

    await T3
    Q = unblock(Q_blocked) # This should be zero-copy since Q_blocked was allocated contiguously
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
        #print(check_result(A, Q, R))
    
if __name__ == "__main__":
    with Parla():
        main()
