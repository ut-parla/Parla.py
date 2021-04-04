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

ROWS = 500000 # Must be >> COLS
COLS = 1000
BLOCK_SIZE = 125000

# Timers
t1_start = 0
t2_start = 0
t3_start = 0
t3_end = 0
start = 0
end = 0

# Not used
@specialized
def qr_block(block):
    #print("CPU")
    return np.linalg.qr(block)

@qr_block.variant(gpu)
def qr_block_gpu(block):
    #print("GPU")
    gpu_Q, gpu_R = cp.linalg.qr(block)
    cpu_Q = cp.asnumpy(gpu_Q)
    cpu_R = cp.asnumpy(gpu_R)
    return cpu_Q, cpu_R

# Not used
@specialized
def matmul_block(block_1, block_2):
    return block_1 @ block_2

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

    # Calculate the number of blocks
    nblocks = (nrows + block_size - 1) // block_size # ceiling division

    # Initialize empty lists to store blocks
    Q1_blocked = [None] * nblocks; # Doesn't need to be contiguous, just views
    R1 = np.empty([nblocks * ncols, ncols]) # Concatenated view
    # Q2 is allocated in t2
    Q = np.empty([nrows, ncols]) # Concatenated view

    # Create tasks to perform qr factorization on each block and store them in lists
    t1_start = time.time()
    T1 = TaskSpace()
    for i in range(nblocks):
        # Get block of A
        A_lower = i * block_size # first row in block, inclusive
        A_upper = (i + 1) * block_size # last row in block, exclusive
        A_block = A[A_lower:A_upper]

        # Block view to store Q1 not needed since it's not contiguous

        # Get block view to store R1
        R1_lower = i * ncols
        R1_upper = (i + 1) * ncols

        @spawn(taskid=T1[i], placement=gpu, memory=(4*A_block.nbytes)) # 4* for scratch space
        def t1():
            #print("t1[", i, "] start", sep='')
            #copy_start = time.time()
            A_block_local = clone_here(A_block)
            #copy_end = time.time()
            Q1_blocked[i], R1[R1_lower:R1_upper] = qr_block(A_block_local)
            #qr_end = time.time()
            #print("t1", i, "copy=", copy_end - copy_start, "qrfac=", qr_end - copy_end)
            #print("t1[", i, "] end", sep='')

    # Perform intermediate qr factorization on R1 to get Q2 and final R
    @spawn(dependencies=T1)
    def t2():
        t2_start = time.time()
        print("t1 time:", t2_start - t1_start)
        #print("t2 start")

        # R here is the final R result
        Q2, R = np.linalg.qr(R1) # TODO Do this recursively or on the GPU if it's slow?

        # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
        # Q2 is currently an (ncols * nblocks) x ncols matrix. Need nblocks of ncols rows each
        t2_end = time.time()
        print("t2 time:", t2_end - t2_start)
        return Q2, R

    Q2, R = await t2
    #print("t2 end")

    t3_start = time.time()
    # Create tasks to perform Q1 @ Q2 matrix multiplication by block
    T3 = TaskSpace()
    for i in range(nblocks):
        # Q1 is already in blocks

        # Get block of Q2
        Q2_lower = i * ncols
        Q2_upper = (i + 1) * ncols
        Q2_block = Q2[Q2_lower:Q2_upper]

        # Get block view to store Q
        Q_lower = i * block_size # first row in block, inclusive
        Q_upper = (i + 1) * block_size # last row in block, exclusive

        @spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=gpu, memory=(4*Q1_blocked[i].nbytes))
        def t3():
            #print("t3[", i, "] start", sep='')
            Q1_block_local = clone_here(Q1_blocked[i])
            Q2_block_local = clone_here(Q2_block)
            Q[Q_lower:Q_upper] = matmul_block(Q1_block_local, Q2_block_local)
            #print("t3[", i, "] end", sep='')

    await T3
    t3_end = time.time()
    print("t3 time:", t3_end - t3_start)
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
        for i in range(6):
            start = time.time()
            Q, R = await tsqr_blocked(A, BLOCK_SIZE)
            end = time.time()
            print("Total time:", end - start)
            print()
            #print(check_result(A, Q, R))
    
if __name__ == "__main__":
    with Parla():
        main()
