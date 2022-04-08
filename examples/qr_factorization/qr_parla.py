# https://arxiv.org/pdf/1301.1071.pdf "Direct TSQR"

import sys
import argparse
import numpy as np
import cupy as cp
from time import perf_counter as time
import os

from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.function_decorators import specialized
from parla.tasks import *
from parla.ldevice import LDeviceSequenceBlocked, PartitionedTensor

# Huge class just for taking tons of timing statistics.
class perfStats:
    def __init__(self, ITERS, NROWS, BLOCK_SIZE):
        self.NTASKS = (NROWS + BLOCK_SIZE - 1) // BLOCK_SIZE # ceiling division
        # Within an iteration, we need H2D, kernel, and D2H times *for each task*
        # We also need total time taken for each stage - this isn't just the sum of the parts, due to latency hiding!
        # Lastly we need to know the number of tasks which ran on GPUs
        self.t1_H2D_tasks = [0] * self.NTASKS
        self.t1_ker_tasks = [0] * self.NTASKS
        self.t1_D2H_tasks = [0] * self.NTASKS
        self.t1_tot = 0
        self.t1_is_GPU_tasks = [False] * self.NTASKS

        self.t2_D2D = 0 # Specifically for puregpu
        self.t2_ker = 0 # Specifically for puregpu
        self.t2_tot = 0

        self.t3_H2D_tasks = [0] * self.NTASKS
        self.t3_ker_tasks = [0] * self.NTASKS
        self.t3_D2H_tasks = [0] * self.NTASKS
        self.t3_tot = 0
        self.t3_is_GPU_tasks = [False] * self.NTASKS

        self.tot_time = 0 # We also record the actual total time

        # Consolidated stats
        self.t1_H2D = 0
        self.t1_ker_CPU = 0
        self.t1_ker_GPU = 0
        self.t1_D2H = 0
        self.t1_GPU_task_count = 0

        self.t3_H2D = 0
        self.t3_ker_CPU = 0
        self.t3_ker_GPU = 0
        self.t3_D2H = 0
        self.t3_GPU_task_count = 0

    # Reset all iteration-specific timers and counters
    def reset(self):
        self.t1_tot = 0
        self.t2_tot = 0
        self.t3_tot = 0
        self.tot_time = 0

        for i in range(self.NTASKS):
            self.t1_H2D_tasks[i] = 0
            self.t1_ker_tasks[i] = 0
            self.t1_D2H_tasks[i] = 0
            self.t3_H2D_tasks[i] = 0
            self.t3_ker_tasks[i] = 0
            self.t3_D2H_tasks[i] = 0
            self.t1_is_GPU_tasks[i] = False
            self.t3_is_GPU_tasks[i] = False

        self.t1_H2D = 0
        self.t1_ker_CPU = 0
        self.t1_ker_GPU = 0
        self.t1_D2H = 0
        self.t1_GPU_task_count = 0

        self.t2_D2D = 0
        self.t2_ker= 0

        self.t3_H2D = 0
        self.t3_ker_CPU = 0
        self.t3_ker_GPU = 0
        self.t3_D2H = 0
        self.t3_GPU_task_count = 0

    def sum_tasks(self, times_list, is_gpu_list, want_gpu_tasks):
        try: average_time = sum(times_list[task] for task in range(self.NTASKS) if is_gpu_list[task] == want_gpu_tasks)
        except ZeroDivisionError: average_time = 0
        return average_time

    # Consolidate the stats just for one iteration
    def consolidate_stats(self):
        self.t1_GPU_task_count = np.count_nonzero(self.t1_is_GPU_tasks)
        self.t3_GPU_task_count = np.count_nonzero(self.t3_is_GPU_tasks)

        self.t1_H2D = self.sum_tasks(self.t1_H2D_tasks, self.t1_is_GPU_tasks, True)
        self.t1_ker_CPU = self.sum_tasks(self.t1_ker_tasks, self.t1_is_GPU_tasks, False)
        self.t1_ker_GPU = self.sum_tasks(self.t1_ker_tasks, self.t1_is_GPU_tasks, True)
        self.t1_D2H = self.sum_tasks(self.t1_D2H_tasks, self.t1_is_GPU_tasks, True)

        self.t3_H2D = self.sum_tasks(self.t3_H2D_tasks, self.t3_is_GPU_tasks, True)
        self.t3_ker_CPU = self.sum_tasks(self.t3_ker_tasks, self.t3_is_GPU_tasks, False)
        self.t3_ker_GPU = self.sum_tasks(self.t3_ker_tasks, self.t3_is_GPU_tasks, True)
        self.t3_D2H = self.sum_tasks(self.t3_D2H_tasks, self.t3_is_GPU_tasks, True)

    # Prints averages and standard deviations
    def print_stats(self, iteration):
        print(f"--- ITERATION {iteration} ---")
        print("t1")
        print("Num GPU tasks:", self.t1_GPU_task_count)
        print("H2D:", self.t1_H2D)
        print("CPU kernels:", self.t1_ker_CPU)
        print("GPU kernels:", self.t1_ker_GPU)
        print("D2H:", self.t1_D2H)
        print("Total:", self.t1_tot)
        print()

        print("t2")
        if PLACEMENT_STRING == 'puregpu':
            print("D2D:", self.t2_D2D)
            print("Kernel:", self.t2_ker)
        print("Total:", self.t2_tot)
        print()
    
        print("t3")
        print("Num GPU tasks:", self.t3_GPU_task_count)
        print("H2D:", self.t3_H2D)
        print("CPU kernels:", self.t3_ker_CPU)
        print("GPU kernels:", self.t3_ker_GPU)
        print("D2H:", self.t3_D2H)
        print("Total:", self.t3_tot)
        print()

        print("Full run total:", self.tot_time)
        print()

    # Prints averages and standard deviations in csv format
    def print_stats_csv(self, iteration):
        print("\"", iteration, "\",", sep='', end='')

        print("\"", self.t1_GPU_task_count, "\",", sep='', end='')
        print("\"", self.t1_H2D, "\",", sep='', end='')
        print("\"", self.t1_ker_CPU, "\",", sep='', end='')
        print("\"", self.t1_ker_GPU, "\",", sep='', end='')
        print("\"", self.t1_D2H, "\",", sep='', end='')
        print("\"", self.t1_tot, "\",", sep='', end='')

        if PLACEMENT_STRING == 'puregpu':
            print("\"", self.t2_D2D, "\",", sep='', end='')
            print("\"", self.t2_ker, "\",", sep='', end='')
        print("\"", self.t2_tot, "\",", sep='', end='')
    
        print("\"", self.t3_GPU_task_count, "\",", sep='', end='')
        print("\"", self.t3_H2D, "\",", sep='', end='')
        print("\"", self.t3_ker_CPU, "\",", sep='', end='')
        print("\"", self.t3_ker_GPU, "\",", sep='', end='')
        print("\"", self.t3_D2H, "\",", sep='', end='')
        print("\"", self.t3_tot, "\",", sep='', end='')

        print("\"", self.tot_time, "\",", sep='')

# CPU QR factorization kernel
@specialized
def qr_block(block, taskid):
    t1_ker_start = time()
    Q, R = np.linalg.qr(block)
    cp.cuda.get_current_stream().synchronize()
    t1_ker_end = time()
    perf_stats.t1_ker_tasks[taskid] = t1_ker_end - t1_ker_start
    return Q, R

# GPU QR factorization kernel and device-to-host transfer
@qr_block.variant(gpu)
def qr_block_gpu(block, taskid):
    perf_stats.t1_is_GPU_tasks[taskid] = True

    # Run the kernel
    t1_ker_start = time()
    gpu_Q, gpu_R = cp.linalg.qr(block)
    cp.cuda.get_current_stream().synchronize()
    t1_ker_end = time()
    perf_stats.t1_ker_tasks[taskid] = t1_ker_end - t1_ker_start

    # Transfer the data
    t1_D2H_start = time()
    cpu_Q = cp.asnumpy(gpu_Q)
    cpu_R = cp.asnumpy(gpu_R)
    cp.cuda.get_current_stream().synchronize()
    t1_D2H_end = time()
    perf_stats.t1_D2H_tasks[taskid] = t1_D2H_end - t1_D2H_start

    return cpu_Q, cpu_R

# CPU matmul kernel
@specialized
def matmul_block(block_1, block_2, taskid):
    t3_ker_start = time()
    Q = block_1 @ block_2
    cp.cuda.get_current_stream().synchronize()
    t3_ker_end = time()
    perf_stats.t3_ker_tasks[taskid] = t3_ker_end - t3_ker_start
    return Q

# GPU matmul kernel and device-to-host transfer
@matmul_block.variant(gpu)
def matmul_block_gpu(block_1, block_2, taskid):
    perf_stats.t3_is_GPU_tasks[taskid] = True

    # Run the kernel
    t3_ker_start = time()
    gpu_Q = cp.matmul(block_1, block_2)
    cp.cuda.get_current_stream().synchronize()
    t3_ker_end = time()
    perf_stats.t3_ker_tasks[taskid] = t3_ker_end - t3_ker_start

    # Transfer the data
    t3_D2H_start = time()
    cpu_Q = cp.asnumpy(gpu_Q)
    cp.cuda.get_current_stream().synchronize()
    t3_D2H_end = time()
    perf_stats.t3_D2H_tasks[taskid] = t3_D2H_end - t3_D2H_start

    return cpu_Q

# A: 2D numpy matrix
# block_size: Positive integer
async def tsqr_blocked(A, block_size):

    nrows, ncols = A.shape

    # Check for block_size > ncols
    assert ncols <= block_size, "Block size must be greater than or equal to the number of columns in the input matrix"

    # Calculate the number of blocks
    nblocks = (nrows + block_size - 1) // block_size # ceiling division
    mapper = LDeviceSequenceBlocked(nblocks, placement=A)
    A_blocked = mapper.partition_tensor(A) # Partition A into blocks

    # Initialize and partition empty array to store blocks (same partitioning scheme, share the mapper)
    Q1_blocked = mapper.partition_tensor(np.empty_like(A))
    R1 = np.empty([nblocks * ncols, ncols]) # Concatenated view
    # Q2 is allocated in t2
    Q = np.empty([nrows, ncols]) # Concatenated view

    # Create tasks to perform qr factorization on each block and store them in lists
    t1_tot_start = time()
    T1 = TaskSpace()
    for i in range(nblocks):
        # Block view to store Q1 not needed since it's not contiguous

        # Get block view to store R1
        R1_lower = i * ncols
        R1_upper = (i + 1) * ncols

        T1_MEMORY = None
        if PLACEMENT_STRING == 'gpu' or PLACEMENT_STRING == 'both':
            T1_MEMORY = int(4.2*A_blocked[i:i+1].nbytes) # Estimate based on empirical evidence

        @spawn(taskid=T1[i], placement=PLACEMENT, memory=T1_MEMORY, vcus=ACUS)
        def t1():
            #print("t1[", i, "] start on ", get_current_devices(), sep='', flush=True)

            # Copy the data to the processor
            t1_H2D_start = time()
            A_block_local = A_blocked[i:i+1]
            cp.cuda.get_current_stream().synchronize()
            t1_H2D_end = time()
            perf_stats.t1_H2D_tasks[i] = t1_H2D_end - t1_H2D_start

            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            Q1_blocked[i], R1[R1_lower:R1_upper] = qr_block(A_block_local, i)

            #print("t1[", i, "] end on ", get_current_devices(),  sep='', flush=True)

    await t1
    t1_tot_end = time()
    perf_stats.t1_tot = t1_tot_end - t1_tot_start

    # Perform intermediate qr factorization on R1 to get Q2 and final R
    t2_tot_start = time()
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
    t2_tot_end = time()
    perf_stats.t2_tot = t2_tot_end - t2_tot_start
    #print("t2 end\n", flush=True)

    # Partition Q2 (same partitioning scheme, share the mapper)
    Q2_blocked = mapper.partition_tensor(Q2)
    t3_tot_start = time()
    # Create tasks to perform Q1 @ Q2 matrix multiplication by block
    T3 = TaskSpace()
    for i in range(nblocks):
        # Q1 is already in blocks

        # Get block view to store Q
        Q_lower = i * block_size # first row in block, inclusive
        Q_upper = (i + 1) * block_size # last row in block, exclusive

        T3_MEMORY = None
        if PLACEMENT_STRING == 'gpu' or PLACEMENT_STRING == 'both':
            T3_MEMORY = 4*Q1_blocked[i].nbytes # # This is a guess

        @spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=PLACEMENT, memory=T3_MEMORY, vcus=ACUS)
        def t3():
            #print("t3[", i, "] start on ", get_current_devices(), sep='', flush=True)

            # Copy the data to the processor
            t3_H2D_start = time()
            Q1_block_local = Q1_blocked[i]
            Q2_block_local = Q2_blocked[i:i+1]
            cp.cuda.get_current_stream().synchronize()
            t3_H2D_end = time()
            perf_stats.t3_H2D_tasks[i] = t3_H2D_end - t3_H2D_start

            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            Q[Q_lower:Q_upper] = matmul_block(Q1_block_local, Q2_block_local, i)

            #print("t3[", i, "] end on ", get_current_devices(), sep='', flush=True)

    await T3
    t3_tot_end = time()
    perf_stats.t3_tot = t3_tot_end - t3_tot_start
    return Q, R

# A: View of matrices on GPU
# block_size: Positive integer
async def tsqr_blocked_puregpu(A, block_size):
    Q1 = [None] * NGPUS
    R1 = PartitionedTensor([None] * NGPUS) # CAVEAT: PartitionedTensor with None holes can be fragile! Be cautious!

    # Create tasks to perform qr factorization on each block and store them in lists
    t1_tot_start = time()
    T1 = TaskSpace()
    for i in range(NGPUS):
        @spawn(taskid=T1[i], placement=A.base[i]) # NB: A[i] dumbly moves the block here!
        def t1():
            #print("t1[", i, "] start on ", get_current_devices(), sep='', flush=True)

            perf_stats.t1_is_GPU_tasks[i] = True
            t1_ker_start = time()
            Q1[i], R1[i] = cp.linalg.qr(A[i])
            R1[i] = R1[i].flatten()
            t1_ker_end = time()
            perf_stats.t1_ker_tasks[i] = t1_ker_end - t1_ker_start
            A[i] = None # Free up memory

            #print("t1[", i, "] end on ", get_current_devices(),  sep='', flush=True)

    await t1
    t1_tot_end = time()
    perf_stats.t1_tot = t1_tot_end - t1_tot_start

    # Perform intermediate qr factorization on R1 to get Q2 and final R
    t2_tot_start = time()
    @spawn(dependencies=T1, placement=gpu)
    def t2():
        #print("\nt2 start", flush=True)

        # Gather to this device
        t2_D2D_start = time()
        R1_reduced = np.empty(shape=(0, NCOLS))
        for dev in range(NGPUS):
            next = R1[dev]
            next = next.reshape(NCOLS, NCOLS)
            R1_reduced = cp.vstack((R1_reduced, next))
            R1[dev] = None # Free up memory
        t2_D2D_end = time()
        perf_stats.t2_D2D = t2_D2D_end - t2_D2D_start

        # R here is the final R result
        t2_ker_start = time()
        Q2, R = cp.linalg.qr(R1_reduced)
        Q2 = Q2.flatten()
        t2_ker_end = time()
        perf_stats.t2_ker = t2_ker_end - t2_ker_start

        return Q2, R

    Q2, R = await t2
    t2_tot_end = time()
    perf_stats.t2_tot = t2_tot_end - t2_tot_start
    #print("t2 end\n", flush=True)

    mapper = LDeviceSequenceBlocked(NGPUS, placement=Q2)
    Q2p = mapper.partition_tensor(Q2)
    Q = [None] * NGPUS
    t3_tot_start = time()
    # Create tasks to perform Q1 @ Q2 matrix multiplication by block
    T3 = TaskSpace()
    for i in range(NGPUS):
        @spawn(taskid=T3[i], dependencies=[T1[i], t2], placement=Q1[i])
        def t3():
            #print("t3[", i, "] start on ", get_current_devices(), sep='', flush=True)
            perf_stats.t3_is_GPU_tasks[i] = True

            # Copy the data to the processor
            # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
            # Q2 is currently an (ncols * nblocks) x ncols matrix. Need nblocks of ncols rows each
            t3_H2D_start = time()
            Q2_local = Q2p[i]
            Q2_local = Q2_local.reshape(NCOLS, NCOLS)
            t3_H2D_end = time()
            perf_stats.t3_H2D_tasks[i] = t3_H2D_end - t3_H2D_start

            # Run the kernel. (Data is copied back within this call; timing annotations are added there)
            t3_ker_start = time()
            Q[i] = cp.matmul(Q1[i], Q2_local)
            t3_ker_end = time()
            perf_stats.t3_ker_tasks[i] = t3_ker_end - t3_ker_start

            #print("t3[", i, "] end on ", get_current_devices(), sep='', flush=True)

    await T3
    t3_tot_end = time()
    perf_stats.t3_tot = t3_tot_end - t3_tot_start
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
    async def test_tsqr_blocked(placement=cpu):
        for i in range(WARMUP + ITERS):
            # Reset all iteration-specific timers and counters
            perf_stats.reset()

            # Original matrix
            np.random.seed(i)
            A = np.random.rand(NROWS, NCOLS)

            if PLACEMENT_STRING == 'puregpu':
                if (NROWS % NGPUS != 0):
                    raise ValueError("Pure GPU version requires NROWS %% NGPUS == 0 (currently %i %% %i)" % (NROWS, NGPUS))

                # Partition matrix on GPUs
                mapper = LDeviceSequenceBlocked(NGPUS, placement=[gpu(dev) for dev in range(NGPUS)])
                A_dev = mapper.partition_tensor(A)
                
                tot_start = time()
                Q_dev, R_dev = await tsqr_blocked_puregpu(A_dev, BLOCK_SIZE)
                tot_end = time()

                # Copy the data back
                if CHECK_RESULT:
                    Q = np.empty(shape=(0, NCOLS))
                    for dev in range(NGPUS):
                        with cp.cuda.Device(dev):
                            Q = np.vstack((Q, cp.asnumpy(Q_dev[dev])))

                    R = cp.asnumpy(R_dev)

            else: # Normal version
                # Run and time the algorithm
                tot_start = time()
                Q, R = await tsqr_blocked(A, BLOCK_SIZE)
                tot_end = time()

            perf_stats.tot_time = tot_end - tot_start

            # Combine task timings into totals for this iteration
            perf_stats.consolidate_stats()

            if (i >= WARMUP):
                iter = i - WARMUP
                if CSV:
                    perf_stats.print_stats_csv(iter)
                else:
                    perf_stats.print_stats(iter)
            
            # Check the results
            if CHECK_RESULT:
                if check_result(A, Q, R):
                    print("\nCorrect result!\n")
                else:
                    print("%***** ERROR: Incorrect final result!!! *****%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rows", help="Number of rows for input matrix; must be >> cols", type=int, default=5000)
    parser.add_argument("-c", "--cols", help="Number of columns for input matrix", type=int, default=100)
    parser.add_argument("-b", "--block_size", help="Block size to break up input matrix; must be >= cols", type=int, default=500)
    parser.add_argument("-i", "--iterations", help="Number of iterations to run experiment. If > 1, first is ignored as warmup.", type=int, default=1)
    parser.add_argument("-w", "--warmup", help="Number of warmup runs to perform before iterations.", type=int, default=0)
    parser.add_argument("-t", "--threads", help="Sets OMP_NUM_THREADS", default='16')
    parser.add_argument("-g", "--ngpus", help="Sets number of GPUs to run on. If set to more than you have, undefined behavior", type=int, default='4')
    parser.add_argument("-p", "--placement", help="'cpu' or 'gpu' or 'both' or 'puregpu'", default='gpu')
    parser.add_argument("-K", "--check_result", help="Checks final result on CPU", action="store_true")
    parser.add_argument("--csv", help="Prints stats in csv format", action="store_true")

    args = parser.parse_args()

    # Set global config variables
    NROWS = args.rows
    NCOLS = args.cols
    BLOCK_SIZE = args.block_size
    ITERS = args.iterations
    WARMUP = args.warmup
    NTHREADS = args.threads
    NGPUS = args.ngpus
    PLACEMENT_STRING = args.placement
    CHECK_RESULT = args.check_result
    CSV = args.csv

    # Set up PLACEMENT variable
    if PLACEMENT_STRING == 'cpu':
        PLACEMENT = cpu
        ACUS = None
    elif PLACEMENT_STRING == 'gpu':
        PLACEMENT = [gpu(i) for i in range(NGPUS)]
        ACUS = None
    elif PLACEMENT_STRING == 'both':
        PLACEMENT = [cpu(0)] + [gpu(i) for i in range(NGPUS)]
        ACUS = 1
    elif PLACEMENT_STRING == 'puregpu':
        PLACEMENT = [gpu(i) for i in range(NGPUS)]
        ACUS = None
        BLOCK_SIZE = int(NROWS / NGPUS)
    else:
        print("Invalid value for placement. Must be 'cpu' or 'gpu' or 'both' or 'puregpu'")

    perf_stats = perfStats(ITERS, NROWS, BLOCK_SIZE)

    print('%**********************************************************************************************%\n')
    print('Config: rows=', NROWS, ' cols=', NCOLS, ' block_size=', BLOCK_SIZE, ' iterations=', ITERS, ' warmup=', WARMUP, \
        ' threads=', NTHREADS, ' ngpus=', NGPUS, ' placement=', PLACEMENT_STRING, ' check_result=', CHECK_RESULT, ' csv=', CSV, sep='')

    if PLACEMENT_STRING == 'puregpu':
        print('puregpu version chosen: block size automatically set to NROWS / NGPUS\n')

    with Parla():
        os.environ['OMP_NUM_THREADS'] = NTHREADS
        main()

    print('%**********************************************************************************************%\n')
