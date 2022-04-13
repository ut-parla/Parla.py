# https://arxiv.org/pdf/1301.1071.pdf "Direct TSQR"

from parla.multiload import multiload_contexts as VECs

import os
import sys
import getopt
import concurrent.futures
import time
import queue

# Default values, can override with command line args
ROWS = 240000 # Must be >> COLS
COLS = 1000
BLOCK_SIZE = 3000 # Must be >= COLS
MAX_WORKERS = 11 # Must be < 12 due to limited VECs
THREADS_PER_WORKER = 6 # For OMP_NUM_THREADS

main_VEC = None
locks = None
VEC_q = queue.Queue()

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
    return np.concatenate([fixarr(a) for a in A])

def VEC_qr(A):
    # Acquire lock
    VEC_id = VEC_q.get()

    mystring = ['|' for x in range(MAX_WORKERS)]
    mystring[VEC_id] = 'x'
    print(mystring)

    with VECs[VEC_id]:
        Q, R = np.linalg.qr(fixarr(A))

    mystring = ['|' for x in range(MAX_WORKERS)]
    mystring[VEC_id] = 'o'
    print(mystring)

    # Release Lock
    VEC_q.task_done()
    VEC_q.put(VEC_id)
    return Q, R

def tsqr_blocked_multi(A, block_size, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers = workers) as executor:
        if COLS > BLOCK_SIZE:
            print('Block size must be greater than or equal to the number of columns in the input matrix', file=sys.stderr)
            exit(1)

        with main_VEC:
            A_blocked, nblocks = make_blocked(A, block_size)

        print('ENTERING PARALLEL SECTION')
        # Each thread gets a block from A_blocked to run numpy's build-in qr factorization on
        block_results = executor.map(VEC_qr, A_blocked)

        # Regroup results
        with main_VEC:
            Q1 = []
            R1 = []
            for result in block_results:
                Q1.append(result[0])
                R1.append(result[1])

            print('FINISHED FIRST PARALLEL SECTION (segfault fixed if this prints)')

def fixarr(A):
    return np.asarray(memoryview(A))

if __name__ == "__main__":

    print('ROWS=', ROWS, ' COLS=', COLS, ' BLOCK_SIZE=', BLOCK_SIZE, ' MAX_WORKERS=', MAX_WORKERS, 'THREADS_PER_WORKER=', THREADS_PER_WORKER, sep='')

    # Set up VEC's
    for i in range(MAX_WORKERS):
        # Limit thread count here
        VECs[i].setenv('OMP_NUM_THREADS', str(THREADS_PER_WORKER))
        with VECs[i]:
            import numpy as np

        # Populate VEC queue
        VEC_q.put(i)
    
    # Reserve last context for single threaded stuff
    # Unlimited threads, don't setenv
    with VECs[MAX_WORKERS]:
        import numpy as np
    main_VEC = VECs[MAX_WORKERS]

    with main_VEC:
        # Original matrix
        A = np.random.rand(ROWS, COLS)

    # Multithreaded blocked version with VECs
    Q, R = tsqr_blocked_multi(A, BLOCK_SIZE, MAX_WORKERS)
