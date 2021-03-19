# https://arxiv.org/pdf/1301.1071.pdf "Direct TSQR"

import sys
import numpy as np
import cupy as cp
import time
import os

from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.array import clone_here
from parla.function_decorators import specialized
from parla.tasks import *

ROWS = 500000 #Must be >> COLS
COLS = 1000
BLOCK_SIZE = 31250
ITERS = 6 # I later through away the first iteration, so really one less
THREADS = '16'

# ************** TIMERS **************
now = time.time

# I need to know the number of blocks up front, because each task has its own timing
NBLOCKS = (ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE # ceiling division

# Within an iteration, we need H2D, kernel, and D2H times *for each task*
t1_H2D_iter = [None] * NBLOCKS
t1_ker_iter = [None] * NBLOCKS
t1_D2H_iter = [None] * NBLOCKS

# t2 is already reduced and only has 1 task

t3_H2D_iter = [None] * NBLOCKS
t3_ker_iter = [None] * NBLOCKS
t3_D2H_iter = [None] * NBLOCKS

# We also need total time taken for each stage - this isn't just the sum of the above, due to latency hiding!
t1_tot_iter = 0
t2_tot_iter = 0
t3_tot_iter = 0

# We'll need to combine an iteration's metrics into totals
t1_H2D_times = [None] * ITERS
t1_ker_times_CPU = [None] * ITERS
t1_ker_times_GPU = [None] * ITERS
t1_D2H_times = [None] * ITERS
t1_tot_times = [None] * ITERS

t2_tot_times = [None] * ITERS

t3_H2D_times = [None] * ITERS
t3_ker_times_CPU = [None] * ITERS
t3_ker_times_GPU = [None] * ITERS
t3_D2H_times = [None] * ITERS
t3_tot_times = [None] * ITERS

# Finally, an actual total time, as a sanity check
tot_times = [None] * ITERS

# Also useful for when doing mixed architecture
t1_is_GPU_iter = [False] * NBLOCKS
t3_is_GPU_iter = [False] * NBLOCKS

t1_GPU_tasks = [None] * ITERS
t3_GPU_tasks = [None] * ITERS

# ************** END TIMERS **************

# CPU QR factorization kernel
@specialized
def qr_block(block, taskid):
    t1_ker_iter_start = now()
    Q, R = np.linalg.qr(block)
    t1_ker_iter_end = now()
    t1_ker_iter[taskid] = t1_ker_iter_end - t1_ker_iter_start
    return Q, R

# GPU QR factorization kernel and device-to-host transfer
@qr_block.variant(gpu)
def qr_block_gpu(block, taskid):
    t1_is_GPU_iter[taskid] = True

    # Run the kernel
    t1_ker_iter_start = now()
    gpu_Q, gpu_R = cp.linalg.qr(block)
    t1_ker_iter_end = now()
    t1_ker_iter[taskid] = t1_ker_iter_end - t1_ker_iter_start

    # Transfer the data
    t1_D2H_iter_start = now()
    cpu_Q = cp.asnumpy(gpu_Q)
    cpu_R = cp.asnumpy(gpu_R)
    t1_D2H_iter_end = now()
    t1_D2H_iter[taskid] = t1_D2H_iter_end - t1_D2H_iter_start

    return cpu_Q, cpu_R

# CPU matmul kernel
@specialized
def matmul_block(block_1, block_2, taskid):
    t3_ker_iter_start = now()
    Q = block_1 @ block_2
    t3_ker_iter_end = now()
    t3_ker_iter[taskid] = t3_ker_iter_end - t3_ker_iter_start
    return Q

# GPU matmul kernel and device-to-host transfer
@matmul_block.variant(gpu)
def matmul_block_gpu(block_1, block_2, taskid):
    t3_is_GPU_iter[taskid] = True

    # Run the kernel
    t3_ker_iter_start = now()
    gpu_Q = cp.matmul(block_1, block_2)
    t3_ker_iter_end = now()
    t3_ker_iter[taskid] = t3_ker_iter_end - t3_ker_iter_start

    # Transfer the data
    t3_D2H_iter_start = now()
    cpu_Q = cp.asnumpy(gpu_Q)
    t3_D2H_iter_end = now()
    t3_D2H_iter[taskid] = t3_D2H_iter_end - t3_D2H_iter_start

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
    t1_tot_iter_start = now()
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
        #@spawn(taskid=T1[i], placement=cpu)
        #@spawn(taskid=T1[i], placement=(cpu, gpu), memory=(4*A_block.nbytes)) # 4* for scratch space
        def t1():
            #print("t1[", i, "] start", sep='', flush=True)

            # Copy the data to the processor
            t1_H2D_iter_start = now()
            A_block_local = clone_here(A_block)
            t1_H2D_iter_end = now()
            t1_H2D_iter[i] = t1_H2D_iter_end - t1_H2D_iter_start

            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            Q1_blocked[i], R1[R1_lower:R1_upper] = qr_block(A_block_local, i)

            #print("t1[", i, "] end", sep='', flush=True)

    await t1
    t1_tot_iter_end = now()
    global t1_tot_iter
    t1_tot_iter = t1_tot_iter_end - t1_tot_iter_start

    # Perform intermediate qr factorization on R1 to get Q2 and final R
    t2_tot_iter_start = now()
    @spawn(dependencies=T1)
    def t2():
        #print("t2 start", flush=True)

        # R here is the final R result
        Q2, R = np.linalg.qr(R1) # TODO Do this recursively or on the GPU if it's slow?

        # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
        # Q2 is currently an (ncols * nblocks) x ncols matrix. Need nblocks of ncols rows each
        return Q2, R

    Q2, R = await t2
    t2_tot_iter_end = now()
    global t2_tot_iter
    t2_tot_iter = t2_tot_iter_end - t2_tot_iter_start
    #print("t2 end", flush=True)

    t3_tot_iter_start = now()
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
        #@spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=cpu)
        #@spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=(cpu, gpu), memory=(4*Q1_blocked[i].nbytes))
        def t3():
            #print("t3[", i, "] start", sep='', flush=True)

            # Copy the data to the processor
            t3_H2D_iter_start = now()
            Q1_block_local = clone_here(Q1_blocked[i])
            Q2_block_local = clone_here(Q2_block)
            t3_H2D_iter_end = now()
            t3_H2D_iter[i] = t3_H2D_iter_end - t3_H2D_iter_start

            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            Q[Q_lower:Q_upper] = matmul_block(Q1_block_local, Q2_block_local, i)

            #print("t3[", i, "] end", sep='', flush=True)

    await T3
    t3_tot_iter_end = now()
    global t3_tot_iter
    t3_tot_iter = t3_tot_iter_end - t3_tot_iter_start
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
        global t1_tot_iter
        global t2_tot_iter

        global t3_tot_iter
        global t1_GPU_tasks
        global t1_H2D_times
        global t1_ker_times_CPU
        global t1_ker_times_GPU
        global t1_D2H_times
        global t1_tot_times

        global t2_tot_times

        global t3_GPU_tasks
        global t3_H2D_times
        global t3_ker_times_CPU
        global t3_ker_times_GPU
        global t3_D2H_times
        global t3_tot_times

        global tot_times

        for i in range(ITERS):
            # Reset all iteration-specific timers and counters
            t1_tot_iter = 0
            t2_tot_iter = 0
            t3_tot_iter = 0

            for j in range(NBLOCKS):
                t1_H2D_iter[j] = None
                t1_ker_iter[j] = None
                t1_D2H_iter[j] = None
                t3_H2D_iter[j] = None
                t3_ker_iter[j] = None
                t3_D2H_iter[j] = None
                t1_is_GPU_iter[j] = False
                t3_is_GPU_iter[j] = False

            # Original matrix
            np.random.seed(i)
            A = np.random.rand(ROWS, COLS)
        
            # Run and time the algorithm
            tot_start = now()
            Q, R = await tsqr_blocked(A, BLOCK_SIZE)
            tot_end = now()
            tot_times[i] = tot_end - tot_start

            # Figure out how many tasks ran on GPU
            t1_GPU_tasks[i] = np.count_nonzero(t1_is_GPU_iter)
            t3_GPU_tasks[i] = np.count_nonzero(t3_is_GPU_iter)

            # Update all timers
            try: t1_H2D_times[i] = sum(t1_H2D_iter[task] for task in range(NBLOCKS) if t1_is_GPU_iter[task] == True) / t1_GPU_tasks[i]
            except ZeroDivisionError: t1_H2D_times[i] = 0

            try: t1_ker_times_CPU[i] = sum(t1_ker_iter[task] for task in range(NBLOCKS) if t1_is_GPU_iter[task] == False) / (NBLOCKS - t1_GPU_tasks[i])
            except ZeroDivisionError: t1_ker_times_CPU[i] = 0

            try: t1_ker_times_GPU[i] = sum(t1_ker_iter[task] for task in range(NBLOCKS) if t1_is_GPU_iter[task] == True) / t1_GPU_tasks[i]
            except ZeroDivisionError: t1_ker_times_GPU[i] = 0

            try: t1_D2H_times[i] = sum(t1_D2H_iter[task] for task in range(NBLOCKS) if t1_is_GPU_iter[task] == True) / t1_GPU_tasks[i]
            except ZeroDivisionError: t1_D2H_times[i] = 0

            t1_tot_times[i] = t1_tot_iter
            
            t2_tot_times[i] = t2_tot_iter
            
            try: t3_H2D_times[i] = sum(t3_H2D_iter[task] for task in range(NBLOCKS) if t3_is_GPU_iter[task] == True) / t3_GPU_tasks[i]
            except ZeroDivisionError: t3_H2D_times[i] = 0

            try: t3_ker_times_CPU[i] = sum(t3_ker_iter[task] for task in range(NBLOCKS) if t3_is_GPU_iter[task] == False) / (NBLOCKS - t3_GPU_tasks[i])
            except ZeroDivisionError: t3_ker_times_CPU[i] = 0

            try: t3_ker_times_GPU[i] = sum(t3_ker_iter[task] for task in range(NBLOCKS) if t3_is_GPU_iter[task] == True) / t3_GPU_tasks[i]
            except ZeroDivisionError: t3_ker_times_GPU[i] = 0

            try: t3_D2H_times[i] = sum(t3_D2H_iter[task] for task in range(NBLOCKS) if t3_is_GPU_iter[task] == True) / t3_GPU_tasks[i]
            except ZeroDivisionError: t3_D2H_times[i] = 0

            t3_tot_times[i] = t3_tot_iter
            
            # Check the results if you want
            #print(check_result(A, Q, R))

        # Cut out the first iteration
        if (ITERS > 1):
            t1_GPU_tasks = t1_GPU_tasks[1:]
            t1_H2D_times = t1_H2D_times[1:]
            t1_ker_times_CPU = t1_ker_times_CPU[1:]
            t1_ker_times_GPU = t1_ker_times_GPU[1:]
            t1_D2H_times = t1_D2H_times[1:]
            t1_tot_times = t1_tot_times[1:]

            t2_tot_times = t2_tot_times[1:]

            t3_GPU_tasks = t3_GPU_tasks[1:]
            t3_H2D_times = t3_H2D_times[1:]
            t3_ker_times_CPU = t3_ker_times_CPU[1:]
            t3_ker_times_GPU = t3_ker_times_GPU[1:]
            t3_D2H_times = t3_D2H_times[1:]
            t3_tot_times = t3_tot_times[1:]

            tot_times = tot_times[1:]

        # Print stuff per iteration
        """
        print("t1 stats per iteration")
        print("Num GPU tasks:", t1_GPU_tasks)
        print("H2D:", t1_H2D_times)
        print("CPU kernels:", t1_ker_times_CPU)
        print("GPU kernels:", t1_ker_times_GPU)
        print("D2H:", t1_D2H_times)
        print("Total:", t1_tot_times)
        print()

        print("t2 stats per iteration")
        print("Total:", t2_tot_times)
        print()
    
        print("t3 stats per iteration")
        print("Num GPU tasks:", t3_GPU_tasks)
        print("H2D:", t3_H2D_times)
        print("CPU kernels:", t3_ker_times_CPU)
        print("GPU kernels:", t3_ker_times_GPU)
        print("D2H:", t3_D2H_times)
        print("Total:", t3_tot_times)
        print()

        print("Full run stats per iteration")
        print("Total:", tot_times)
        print()
        """

        # Get averages across iterations
        t1_GPU_tasks_avg = np.average(t1_GPU_tasks)

        t1_GPU_tasks_std = np.std(t1_GPU_tasks)

        try: t1_H2D_times_avg = np.average(t1_H2D_times, weights=t1_GPU_tasks)
        except ZeroDivisionError: t1_H2D_times_avg = 0
        
        try: t1_ker_times_CPU_avg = np.average(t1_ker_times_CPU, weights=[NBLOCKS - nGPU for nGPU in t1_GPU_tasks])
        except ZeroDivisionError: t1_ker_times_CPU_avg = 0

        try: t1_ker_times_GPU_avg = np.average(t1_ker_times_GPU, weights=t1_GPU_tasks)
        except ZeroDivisionError: t1_ker_times_GPU_avg = 0

        try: t1_D2H_times_avg = np.average(t1_D2H_times, weights=t1_GPU_tasks)
        except ZeroDivisionError: t1_D2H_times_avg = 0

        t1_tot_times_avg = np.average(t1_tot_times)

        t1_tot_times_std = np.std(t1_tot_times)

        t2_tot_times_avg = np.average(t2_tot_times)

        t2_tot_times_std = np.std(t2_tot_times)

        t3_GPU_tasks_avg = np.average(t3_GPU_tasks)

        t3_GPU_tasks_std = np.std(t3_GPU_tasks)

        try: t3_H2D_times_avg = np.average(t3_H2D_times, weights=t3_GPU_tasks)
        except ZeroDivisionError: t3_H2D_times_avg = 0

        try: t3_ker_times_CPU_avg = np.average(t3_ker_times_CPU, weights=[NBLOCKS - nGPU for nGPU in t3_GPU_tasks])
        except ZeroDivisionError: t3_ker_times_CPU_avg = 0

        try: t3_ker_times_GPU_avg = np.average(t3_ker_times_GPU, weights=t3_GPU_tasks)
        except ZeroDivisionError: t3_ker_times_GPU_avg = 0

        try: t3_D2H_times_avg = np.average(t3_D2H_times, weights=t3_GPU_tasks)
        except ZeroDivisionError: t3_D2H_times_avg = 0

        t3_tot_times_avg = np.average(t3_tot_times)

        t3_tot_times_std = np.std(t3_tot_times)

        tot_times_avg = np.average(tot_times)

        tot_times_std = np.std(tot_times)

        # Print average stats over all iterations
        print("t1 averages")
        print("Num GPU tasks:", t1_GPU_tasks_avg, "±", t1_GPU_tasks_std)
        print("H2D:", t1_H2D_times_avg)
        print("CPU kernels:", t1_ker_times_CPU_avg)
        print("GPU kernels:", t1_ker_times_GPU_avg)
        print("D2H:", t1_D2H_times_avg)
        print("Total:", t1_tot_times_avg, "±", t1_tot_times_std)
        print()

        print("t2 averages")
        print("Total:", t2_tot_times_avg, "±", t2_tot_times_std)
        print()
    
        print("t3 averages")
        print("Num GPU tasks:", t3_GPU_tasks_avg, "±", t3_GPU_tasks_std)
        print("H2D:", t3_H2D_times_avg)
        print("CPU kernels:", t3_ker_times_CPU_avg)
        print("GPU kernels:", t3_ker_times_GPU_avg)
        print("D2H:", t3_D2H_times_avg)
        print("Total:", t3_tot_times_avg, "±", t3_tot_times_std)
        print()

        print("Full run averages")
        print("Total:", tot_times_avg, "±", tot_times_std)
        print()

if __name__ == "__main__":
    with Parla():
        os.environ['OMP_NUM_THREADS'] = THREADS
        main()
