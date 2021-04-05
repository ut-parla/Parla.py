# https://arxiv.org/pdf/1301.1071.pdf "Direct TSQR"

import sys
import argparse
import numpy as np
import cupy as cp
from time import time
import os

from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.array import clone_here
from parla.function_decorators import specialized
from parla.tasks import *

# Huge class just for taking tons of timing statistics.
class perfStats:
    def __init__(self, ITERS, NROWS, BLOCK_SIZE):
        self.NTASKS = (NROWS + BLOCK_SIZE - 1) // BLOCK_SIZE # ceiling division
        # Within an iteration, we need H2D, kernel, and D2H times *for each task*
        # We also need total time taken for each stage - this isn't just the sum of the parts, due to latency hiding!
        # Lastly we need to know the number of tasks which ran on GPUs
        self.t1_H2D_iter = [None] * self.NTASKS
        self.t1_ker_iter = [None] * self.NTASKS
        self.t1_D2H_iter = [None] * self.NTASKS
        self.t1_tot_iter = 0
        self.t1_is_GPU_iter = [False] * self.NTASKS

        self.t2_tot_iter = 0 # t2 is already reduced and only has 1 task and only runs on the CPU

        self.t3_H2D_iter = [None] * self.NTASKS
        self.t3_ker_iter = [None] * self.NTASKS
        self.t3_D2H_iter = [None] * self.NTASKS
        self.t3_tot_iter = 0
        self.t3_is_GPU_iter = [False] * self.NTASKS

        # We'll need to combine an iteration's metrics into totals
        self.t1_H2D_times = [None] * ITERS
        self.t1_ker_times_CPU = [None] * ITERS
        self.t1_ker_times_GPU = [None] * ITERS
        self.t1_D2H_times = [None] * ITERS
        self.t1_tot_times = [None] * ITERS
        self.t1_GPU_tasks = [None] * ITERS

        self.t2_tot_times = [None] * ITERS

        self.t3_H2D_times = [None] * ITERS
        self.t3_ker_times_CPU = [None] * ITERS
        self.t3_ker_times_GPU = [None] * ITERS
        self.t3_D2H_times = [None] * ITERS
        self.t3_tot_times = [None] * ITERS
        self.t3_GPU_tasks = [None] * ITERS

        self.tot_times = [None] * ITERS # We also record the actual total time

        # And at the end, we'll combine everything into averages and standard deviations
        self.t1_GPU_tasks_avg = 0
        self.t1_GPU_tasks_std = 0
        self.t1_H2D_times_avg = 0
        self.t1_ker_times_CPU_avg = 0
        self.t1_ker_times_GPU_avg = 0
        self.t1_D2H_times_avg = 0
        self.t1_tot_times_avg = 0
        self.t1_tot_times_std = 0

        self.t2_tot_times_avg = 0
        self.t2_tot_times_std = 0

        self.t3_GPU_tasks_avg = 0
        self.t3_GPU_tasks_std = 0
        self.t3_H2D_times_avg = 0
        self.t3_ker_times_CPU_avg = 0
        self.t3_ker_times_GPU_avg = 0
        self.t3_D2H_times_avg = 0
        self.t3_tot_times_avg = 0
        self.t3_tot_times_std = 0

        self.tot_times_avg = 0
        self.tot_times_std = 0

    # Reset all iteration-specific timers and counters
    def reset_iter(self):
        self.t1_tot_iter = 0
        self.t2_tot_iter = 0
        self.t3_tot_iter = 0

        for i in range(self.NTASKS):
            self.t1_H2D_iter[i] = None
            self.t1_ker_iter[i] = None
            self.t1_D2H_iter[i] = None
            self.t3_H2D_iter[i] = None
            self.t3_ker_iter[i] = None
            self.t3_D2H_iter[i] = None
            self.t1_is_GPU_iter[i] = False
            self.t3_is_GPU_iter[i] = False

    def average_tasks(self, times_list, is_gpu_list, want_gpu_tasks, ntasks):
        try: average_time = sum(times_list[task] for task in range(self.NTASKS) if is_gpu_list[task] == want_gpu_tasks) / ntasks
        except ZeroDivisionError: average_time = 0
        return average_time

    # Consolidate the stats just for one iteration
    def consolidate_iter_stats(self, iter):
        self.t1_GPU_tasks[iter] = np.count_nonzero(self.t1_is_GPU_iter)
        self.t3_GPU_tasks[iter] = np.count_nonzero(self.t3_is_GPU_iter)

        self.t1_H2D_times[iter] = self.average_tasks(self.t1_H2D_iter, self.t1_is_GPU_iter, True, self.t1_GPU_tasks[iter])
        self.t1_ker_times_CPU[iter] = self.average_tasks(self.t1_ker_iter, self.t1_is_GPU_iter, False, (self.NTASKS - self.t1_GPU_tasks[iter]))
        self.t1_ker_times_GPU[iter] = self.average_tasks(self.t1_ker_iter, self.t1_is_GPU_iter, True, self.t1_GPU_tasks[iter])
        self.t1_D2H_times[iter] = self.average_tasks(self.t1_D2H_iter, self.t1_is_GPU_iter, True, self.t1_GPU_tasks[iter])
        self.t1_tot_times[iter] = self.t1_tot_iter
        
        self.t2_tot_times[iter] = self.t2_tot_iter

        self.t3_H2D_times[iter] = self.average_tasks(self.t3_H2D_iter, self.t3_is_GPU_iter, True, self.t3_GPU_tasks[iter])
        self.t3_ker_times_CPU[iter] = self.average_tasks(self.t3_ker_iter, self.t3_is_GPU_iter, False, (self.NTASKS - self.t3_GPU_tasks[iter]))
        self.t3_ker_times_GPU[iter] = self.average_tasks(self.t3_ker_iter, self.t3_is_GPU_iter, True, self.t3_GPU_tasks[iter])
        self.t3_D2H_times[iter] = self.average_tasks(self.t3_D2H_iter, self.t3_is_GPU_iter, True, self.t3_GPU_tasks[iter])
        self.t3_tot_times[iter] = self.t3_tot_iter

    def avg_weighted(self, list_to_avg, weights):
        try: avg = np.average(list_to_avg, weights=weights)
        except ZeroDivisionError: avg = 0
        return avg

    # Consolidate the stats across iterations
    # Note: Taking an average of averages here is OK because I weight them properly
    # This doesn't work with standard deviation though, so I'm just not doing that except for the totals...
    def consolidate_stats(self):
        self.t1_GPU_tasks_avg = np.average(self.t1_GPU_tasks)
        self.t1_GPU_tasks_std = np.std(self.t1_GPU_tasks)
        self.t1_H2D_times_avg = self.avg_weighted(self.t1_H2D_times, weights=self.t1_GPU_tasks)
        self.t1_ker_times_CPU_avg = self.avg_weighted(self.t1_ker_times_CPU, weights=[self.NTASKS - gpu_tasks for gpu_tasks in self.t1_GPU_tasks])
        self.t1_ker_times_GPU_avg = self.avg_weighted(self.t1_ker_times_GPU, weights=self.t1_GPU_tasks)
        self.t1_D2H_times_avg = self.avg_weighted(self.t1_D2H_times, weights=self.t1_GPU_tasks)
        self.t1_tot_times_avg = np.average(self.t1_tot_times)
        self.t1_tot_times_std = np.std(self.t1_tot_times)

        self.t2_tot_times_avg = np.average(self.t2_tot_times)
        self.t2_tot_times_std = np.std(self.t2_tot_times)

        self.t3_GPU_tasks_avg = np.average(self.t3_GPU_tasks)
        self.t3_GPU_tasks_std = np.std(self.t3_GPU_tasks)
        self.t3_H2D_times_avg = self.avg_weighted(self.t3_H2D_times, weights=self.t3_GPU_tasks)
        self.t3_ker_times_CPU_avg = self.avg_weighted(self.t3_ker_times_CPU, weights=[self.NTASKS - gpu_tasks for gpu_tasks in self.t3_GPU_tasks])
        self.t3_ker_times_GPU_avg = self.avg_weighted(self.t3_ker_times_GPU, weights=self.t3_GPU_tasks)
        self.t3_D2H_times_avg = self.avg_weighted(self.t3_D2H_times, weights=self.t3_GPU_tasks)
        self.t3_tot_times_avg = np.average(self.t3_tot_times)
        self.t3_tot_times_std = np.std(self.t3_tot_times)

        self.tot_times_avg = np.average(self.tot_times)
        self.tot_times_std = np.std(self.tot_times)

    def remove_warmup(self):
        self.t1_GPU_tasks = self.t1_GPU_tasks[1:]
        self.t1_H2D_times = self.t1_H2D_times[1:]
        self.t1_ker_times_CPU = self.t1_ker_times_CPU[1:]
        self.t1_ker_times_GPU = self.t1_ker_times_GPU[1:]
        self.t1_D2H_times = self.t1_D2H_times[1:]
        self.t1_tot_times = self.t1_tot_times[1:]

        self.t2_tot_times = self.t2_tot_times[1:]

        self.t3_GPU_tasks = self.t3_GPU_tasks[1:]
        self.t3_H2D_times = self.t3_H2D_times[1:]
        self.t3_ker_times_CPU = self.t3_ker_times_CPU[1:]
        self.t3_ker_times_GPU = self.t3_ker_times_GPU[1:]
        self.t3_D2H_times = self.t3_D2H_times[1:]
        self.t3_tot_times = self.t3_tot_times[1:]

        self.tot_times = self.tot_times[1:]

    # Prints averages and standard deviations
    def print_stats(self):
        print("t1 averages")
        print("Num GPU tasks:", self.t1_GPU_tasks_avg, "±", self.t1_GPU_tasks_std)
        print("H2D:", self.t1_H2D_times_avg)
        print("CPU kernels:", self.t1_ker_times_CPU_avg)
        print("GPU kernels:", self.t1_ker_times_GPU_avg)
        print("D2H:", self.t1_D2H_times_avg)
        print("Total:", self.t1_tot_times_avg, "±", self.t1_tot_times_std)
        print()

        print("t2 averages")
        print("Total:", self.t2_tot_times_avg, "±", self.t2_tot_times_std)
        print()
    
        print("t3 averages")
        print("Num GPU tasks:", self.t3_GPU_tasks_avg, "±", self.t3_GPU_tasks_std)
        print("H2D:", self.t3_H2D_times_avg)
        print("CPU kernels:", self.t3_ker_times_CPU_avg)
        print("GPU kernels:", self.t3_ker_times_GPU_avg)
        print("D2H:", self.t3_D2H_times_avg)
        print("Total:", self.t3_tot_times_avg, "±", self.t3_tot_times_std)
        print()

        print("Full run averages")
        print("Total:", self.tot_times_avg, "±", self.tot_times_std)
        print()

    # Prints averages and standard deviations in csv format
    def print_stats_csv(self):
        print("\"", self.t1_GPU_tasks_avg, "\",", sep='', end='')
        print("\"", self.t1_GPU_tasks_std, "\",", sep='', end='')
        print("\"", self.t1_H2D_times_avg, "\",", sep='', end='')
        print("\"", self.t1_ker_times_CPU_avg, "\",", sep='', end='')
        print("\"", self.t1_ker_times_GPU_avg, "\",", sep='', end='')
        print("\"", self.t1_D2H_times_avg, "\",", sep='', end='')
        print("\"", self.t1_tot_times_avg, "\",", sep='', end='')
        print("\"", self.t1_tot_times_std, "\",", sep='', end='')

        print("\"", self.t2_tot_times_avg, "\",", sep='', end='')
        print("\"", self.t2_tot_times_std, "\",", sep='', end='')
    
        print("\"", self.t3_GPU_tasks_avg, "\",", sep='', end='')
        print("\"", self.t3_GPU_tasks_std, "\",", sep='', end='')
        print("\"", self.t3_H2D_times_avg, "\",", sep='', end='')
        print("\"", self.t3_ker_times_CPU_avg, "\",", sep='', end='')
        print("\"", self.t3_ker_times_GPU_avg, "\",", sep='', end='')
        print("\"", self.t3_D2H_times_avg, "\",", sep='', end='')
        print("\"", self.t3_tot_times_avg, "\",", sep='', end='')
        print("\"", self.t3_tot_times_std, "\",", sep='', end='')

        print("\"", self.tot_times_avg, "\",", sep='', end='')
        print("\"", self.tot_times_std, "\"", sep='')

    # Prints stats for every iteration
    def print_stats_verbose(self):
        print("t1 stats per iteration")
        print("Num GPU tasks:", self.t1_GPU_tasks)
        print("H2D:", self.t1_H2D_times)
        print("CPU kernels:", self.t1_ker_times_CPU)
        print("GPU kernels:", self.t1_ker_times_GPU)
        print("D2H:", self.t1_D2H_times)
        print("Total:", self.t1_tot_times)
        print()

        print("t2 stats per iteration")
        print("Total:", self.t2_tot_times)
        print()
    
        print("t3 stats per iteration")
        print("Num GPU tasks:", self.t3_GPU_tasks)
        print("H2D:", self.t3_H2D_times)
        print("CPU kernels:", self.t3_ker_times_CPU)
        print("GPU kernels:", self.t3_ker_times_GPU)
        print("D2H:", self.t3_D2H_times)
        print("Total:", self.t3_tot_times)
        print()

        print("Full run stats per iteration")
        print("Total:", self.tot_times)
        print()

# CPU QR factorization kernel
@specialized
def qr_block(block, taskid):
    t1_ker_iter_start = time()
    Q, R = np.linalg.qr(block)
    t1_ker_iter_end = time()
    perf_stats.t1_ker_iter[taskid] = t1_ker_iter_end - t1_ker_iter_start
    return Q, R

# GPU QR factorization kernel and device-to-host transfer
@qr_block.variant(gpu)
def qr_block_gpu(block, taskid):
    perf_stats.t1_is_GPU_iter[taskid] = True

    # Run the kernel
    t1_ker_iter_start = time()
    gpu_Q, gpu_R = cp.linalg.qr(block)
    t1_ker_iter_end = time()
    perf_stats.t1_ker_iter[taskid] = t1_ker_iter_end - t1_ker_iter_start

    # Transfer the data
    t1_D2H_iter_start = time()
    cpu_Q = cp.asnumpy(gpu_Q)
    cpu_R = cp.asnumpy(gpu_R)
    t1_D2H_iter_end = time()
    perf_stats.t1_D2H_iter[taskid] = t1_D2H_iter_end - t1_D2H_iter_start

    return cpu_Q, cpu_R

# CPU matmul kernel
@specialized
def matmul_block(block_1, block_2, taskid):
    t3_ker_iter_start = time()
    Q = block_1 @ block_2
    t3_ker_iter_end = time()
    perf_stats.t3_ker_iter[taskid] = t3_ker_iter_end - t3_ker_iter_start
    return Q

# GPU matmul kernel and device-to-host transfer
@matmul_block.variant(gpu)
def matmul_block_gpu(block_1, block_2, taskid):
    perf_stats.t3_is_GPU_iter[taskid] = True

    # Run the kernel
    t3_ker_iter_start = time()
    gpu_Q = cp.matmul(block_1, block_2)
    t3_ker_iter_end = time()
    perf_stats.t3_ker_iter[taskid] = t3_ker_iter_end - t3_ker_iter_start

    # Transfer the data
    t3_D2H_iter_start = time()
    cpu_Q = cp.asnumpy(gpu_Q)
    t3_D2H_iter_end = time()
    perf_stats.t3_D2H_iter[taskid] = t3_D2H_iter_end - t3_D2H_iter_start

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
    t1_tot_iter_start = time()
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

        T1_MEMORY = None
        if PLACEMENT_STRING == 'gpu' or PLACEMENT_STRING == 'both':
            T1_MEMORY = int(4.2*A_block.nbytes) # Estimate based on empirical evidence

        @spawn(taskid=T1[i], placement=PLACEMENT, memory=T1_MEMORY)
        def t1():
            #print("t1[", i, "] start on ", get_current_devices(), sep='', flush=True)

            # Copy the data to the processor
            t1_H2D_iter_start = time()
            A_block_local = clone_here(A_block)
            t1_H2D_iter_end = time()
            perf_stats.t1_H2D_iter[i] = t1_H2D_iter_end - t1_H2D_iter_start

            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            Q1_blocked[i], R1[R1_lower:R1_upper] = qr_block(A_block_local, i)

            #print("t1[", i, "] end on ", get_current_devices(),  sep='', flush=True)

    await t1
    t1_tot_iter_end = time()
    perf_stats.t1_tot_iter = t1_tot_iter_end - t1_tot_iter_start

    # Perform intermediate qr factorization on R1 to get Q2 and final R
    t2_tot_iter_start = time()
    @spawn(dependencies=T1, placement=cpu)
    def t2():
        #print("\nt2 start", flush=True)

        # R here is the final R result
        # This step could be done recursively, but for small column counts that's not necessary
        Q2, R = np.linalg.qr(R1)

        # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
        # Q2 is currently an (ncols * nblocks) x ncols matrix. Need nblocks of ncols rows each
        return Q2, R

    Q2, R = await t2
    t2_tot_iter_end = time()
    perf_stats.t2_tot_iter = t2_tot_iter_end - t2_tot_iter_start
    #print("t2 end\n", flush=True)

    t3_tot_iter_start = time()
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

        T3_MEMORY = None
        if PLACEMENT_STRING == 'gpu' or PLACEMENT_STRING == 'both':
            T3_MEMORY = 4*Q1_blocked[i].nbytes # # This is a guess

        @spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=PLACEMENT, memory=T3_MEMORY)
        def t3():
            #print("t3[", i, "] start on ", get_current_devices(), sep='', flush=True)

            # Copy the data to the processor
            t3_H2D_iter_start = time()
            Q1_block_local = clone_here(Q1_blocked[i])
            Q2_block_local = clone_here(Q2_block)
            t3_H2D_iter_end = time()
            perf_stats.t3_H2D_iter[i] = t3_H2D_iter_end - t3_H2D_iter_start

            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            Q[Q_lower:Q_upper] = matmul_block(Q1_block_local, Q2_block_local, i)

            #print("t3[", i, "] end on ", get_current_devices(), sep='', flush=True)

    await T3
    t3_tot_iter_end = time()
    perf_stats.t3_tot_iter = t3_tot_iter_end - t3_tot_iter_start
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

def main():
    @spawn()
    async def test_tsqr_blocked():
        for i in range(ITERS):
            # Reset all iteration-specific timers and counters
            perf_stats.reset_iter()

            # Original matrix
            np.random.seed(i)
            A = np.random.rand(NROWS, NCOLS)
        
            # Run and time the algorithm
            tot_start = time()
            Q, R = await tsqr_blocked(A, BLOCK_SIZE)
            tot_end = time()
            perf_stats.tot_times[i] = tot_end - tot_start

            # Combine task timings into totals for this iteration
            perf_stats.consolidate_iter_stats(i)
            
            # Check the results
            if CHECK_RESULT:
                if check_result(A, Q, R):
                    print("\nCorrect result!\n")
                else:
                    print("%***** ERROR: Incorrect final result!!! *****%")

        # Cut out the warmup iteration (unless we only ran 1)
        if (ITERS > 1):
            perf_stats.remove_warmup()

        # Get averages and standard deviations across iterations
        perf_stats.consolidate_stats()

        # Print out the stats you want
        if PRINT_VERBOSE:
            perf_stats.print_stats_verbose() # Prints per-iteration stats
        if CSV:
            perf_stats.print_stats_csv() # Prints averages and standard deviations in csv format
        else:
            perf_stats.print_stats() # Prints averages and standard deviations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rows", help="Number of rows for input matrix; must be >> cols", type=int, default=5000)
    parser.add_argument("-c", "--cols", help="Number of columns for input matrix", type=int, default=100)
    parser.add_argument("-b", "--block_size", help="Block size to break up input matrix; must be >= cols", type=int, default=500)
    parser.add_argument("-i", "--iterations", help="Number of iterations to run experiment. If > 1, first is ignored as warmup.", type=int, default=1)
    parser.add_argument("-t", "--threads", help="Sets OMP_NUM_THREADS", default='16')
    parser.add_argument("-g", "--ngpus", help="Sets number of GPUs to run on. If set to more than you have, undefined behavior", type=int, default='4')
    parser.add_argument("-p", "--placement", help="'cpu' or 'gpu' or 'both'", default='gpu')
    parser.add_argument("-K", "--check_result", help="Checks final result on CPU", action="store_true")
    parser.add_argument("-v", "--verbose", help="Prints stats for every iteration", action="store_true")
    parser.add_argument("--csv", help="Prints stats in csv format", action="store_true")

    args = parser.parse_args()

    # Set global config variables
    NROWS = args.rows
    NCOLS = args.cols
    BLOCK_SIZE = args.block_size
    ITERS = args.iterations
    NTHREADS = args.threads
    NGPUS = args.ngpus
    PLACEMENT_STRING = args.placement
    CHECK_RESULT = args.check_result
    PRINT_VERBOSE = args.verbose
    CSV = args.csv

    perf_stats = perfStats(ITERS, NROWS, BLOCK_SIZE)

    if not CSV:
        print('%**********************************************************************************************%\n')
        print('Config: rows=', NROWS, ' cols=', NCOLS, ' block_size=', BLOCK_SIZE, ' iterations=', ITERS, ' threads=', NTHREADS, \
            ' ngpus=', NGPUS, ' placement=', PLACEMENT_STRING, ' check_result=', CHECK_RESULT, ' verbose=', PRINT_VERBOSE, ' csv=', CSV, \
            sep='', end='\n\n')

    # Set up PLACEMENT variable
    if PLACEMENT_STRING == 'cpu':
        PLACEMENT = cpu
    elif PLACEMENT_STRING == 'gpu':
        PLACEMENT = [gpu(i) for i in range(NGPUS)]
    elif PLACEMENT_STRING == 'both':
        PLACEMENT = [cpu] + [gpu(i) for i in range(NGPUS)]
    else:
        print("Invalid value for placement. Must be 'cpu' or 'gpu' or 'both'")

    with Parla():
        os.environ['OMP_NUM_THREADS'] = NTHREADS
        main()

    if not CSV:
        print('%**********************************************************************************************%\n')
