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
MAX_WORKERS = 8 # Must be < 12 due to limited VECs
THREADS_PER_WORKER = 4 # For OMP_NUM_THREADS

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

    """
    mystring = ['|' for x in range(MAX_WORKERS)]
    mystring[VEC_id] = 'x'
    print(mystring)
    """

    with VECs[VEC_id]:
        Q, R = np.linalg.qr(fixarr(A))

    """
    mystring = ['|' for x in range(MAX_WORKERS)]
    mystring[VEC_id] = 'o'
    print(mystring)
    """

    # Release Lock
    VEC_q.task_done()
    VEC_q.put(VEC_id)
    return Q, R

def VEC_matmul(A, B):
    # Acquire lock
    VEC_id = VEC_q.get()

    """
    mystring = ['|' for x in range(MAX_WORKERS)]
    mystring[VEC_id] = 'x'
    print(mystring)
    """

    with VECs[VEC_id]:
        M = np.matmul(fixarr(A), fixarr(B))

    """
    mystring = ['|' for x in range(MAX_WORKERS)]
    mystring[VEC_id] = 'o'
    print(mystring)
    """

    # Release Lock
    VEC_q.task_done()
    VEC_q.put(VEC_id)
    return M

def tsqr_blocked_multi(A, block_size, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers = workers) as executor:
        if COLS > BLOCK_SIZE:
            print('Block size must be greater than or equal to the number of columns in the input matrix', file=sys.stderr)
            exit(1)

        with main_VEC:
            A_blocked, nblocks = make_blocked(A, block_size)

        # Parallel step 1
        #print('ENTERING FIRST PARALLEL SECTION')
        # Each thread gets a block from A_blocked to run numpy's build-in qr factorization on
        block_results = executor.map(VEC_qr, A_blocked)

        # Regroup results
        with main_VEC:
            Q1 = []
            R1 = []
            for result in block_results:
                Q1.append(result[0])
                R1.append(result[1])

            #print('FINISHED FIRST PARALLEL SECTION')

            R1 = unblock(R1)

            # R here is the final R result
            Q2, R = np.linalg.qr(R1)

            # Q1 and Q2 must have an equal number of blocks, where Q1 blocks' ncols = Q2 blocks' nrows
            # Q1: block_count = A.nrows / block_size. ncols = A.ncols.
            # Q2: nrows = (A.nrows * A.ncols / block_size). Need block_count = A.nrows / block_size, nrows = A.ncols
            Q2 = make_blocked(Q2, A.shape[1])[0]

        # Parallel step 2
        #print('ENTERING SECOND PARALLEL SECTION')
        # Each thread performs a matrix multiplication call
        Q = executor.map(VEC_matmul, Q1, Q2)

        with main_VEC:
            Q = unblock(Q)
            #print('FINISHED SECOND PARALLEL SECTION')
            return Q, R

def check_result(A, Q, R):
    # Check product
    A = fixarr(A)
    Q = fixarr(Q)
    R = fixarr(R)
    is_correct_prod = np.allclose(np.matmul(Q, R), A)
    
    # Check orthonormal
    Q_check = np.matmul(Q.transpose(), Q)
    is_ortho_Q = np.allclose(Q_check, np.identity(COLS))
    
    # Check upper
    is_upper_R = np.allclose(R, np.triu(R))

    return is_correct_prod and is_ortho_Q and is_upper_R

def fixarr(A):
    return np.asarray(memoryview(A))

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
        elif opt == '-t':
            THREADS_PER_WORKER = int(arg)

    print('ROWS=', ROWS, ' COLS=', COLS, ' BLOCK_SIZE=', BLOCK_SIZE, ' MAX_WORKERS=', MAX_WORKERS, 'THREADS_PER_WORKER=', THREADS_PER_WORKER, sep='')
    
    # Set up VEC's
    for i in range(MAX_WORKERS):
        # Limit thread count here
        VECs[i].setenv('OMP_NUM_THREADS', str(THREADS_PER_WORKER))
        VECs[i].setenv('VECID', str(i))
        with VECs[i]:
            import numpy as np

        # Populate VEC queue
        VEC_q.put(i)
    
    # Reserve last context for single threaded stuff
    # Unlimited threads, don't setenv
    with VECs[MAX_WORKERS]:
        import numpy as np
    main_VEC = VECs[MAX_WORKERS]

    for i in range(6):
        with main_VEC:
            # Original matrix
            A = np.random.rand(ROWS, COLS)

        # Multithreaded blocked version with VECs
        start = time.time()
        Q, R = tsqr_blocked_multi(A, BLOCK_SIZE, MAX_WORKERS)
        end = time.time()
        print(end - start)

        with main_VEC:
            print(check_result(A, Q, R))
