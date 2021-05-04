# https://arxiv.org/pdf/1301.1071.pdf "Direct TSQR"

from parla.multiload import multiload_contexts as VECs

import sys
import argparse
#import mkl # Do this later in the VEC
#import numpy as np # Do this later so we can set the threadpool size
import concurrent.futures
import queue
import time

# Needed because VECs lose track of object type. No copy (I think?)
def fixarr(A):
    return np.asarray(memoryview(A))

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
    mystring = ['|' for x in range(NGROUPS)]
    mystring[VEC_id] = 'x'
    print(mystring)
    """

    with VECs[VEC_id]:
        Q, R = np.linalg.qr(fixarr(A))

    """
    mystring = ['|' for x in range(NGROUPS)]
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
    mystring = ['|' for x in range(NGROUPS)]
    mystring[VEC_id] = 'x'
    print(mystring)
    """

    with VECs[VEC_id]:
        M = np.matmul(fixarr(A), fixarr(B))

    """
    mystring = ['|' for x in range(NGROUPS)]
    mystring[VEC_id] = 'o'
    print(mystring)
    """

    # Release Lock
    VEC_q.task_done()
    VEC_q.put(VEC_id)
    return M

def tsqr_blocked(A):
    with concurrent.futures.ThreadPoolExecutor(max_workers=NGROUPS) as executor:
        if NCOLS > BLOCK_SIZE:
            print('Block size must be greater than or equal to the number of columns in the input matrix', file=sys.stderr)
            exit(1)

        with main_VEC:
            A_blocked, nblocks = make_blocked(A, BLOCK_SIZE)

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
            # Q1: block_count = A.nrows / BLOCK_SIZE. ncols = A.ncols.
            # Q2: nrows = (A.nrows * A.ncols / BLOCK_SIZE). Need block_count = A.nrows / BLOCK_SIZE, nrows = A.ncols
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
    A = fixarr(A)
    Q = fixarr(Q)
    R = fixarr(R)

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
    parser.add_argument("-t", "--threads", help="Number of threads per VEC", default='4')
    parser.add_argument("-g", "--ngroups", help="Number of thread groups to use (max 11)", type=int, default='4')
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
    NGROUPS = args.ngroups
    CHECK_RESULT = args.check_result
    CSV = args.csv

    print('%**********************************************************************************************%\n')
    print('Config: rows=', NROWS, ' cols=', NCOLS, ' block_size=', BLOCK_SIZE, ' iterations=', ITERS, ' warmup=', WARMUP, \
        ' threads=', NTHREADS, ' ngroups=', NGROUPS, ' check_result=', CHECK_RESULT, ' csv=', CSV, sep='')
    
    # Set up VEC's
    VEC_q = queue.Queue()
    for i in range(NGROUPS):
        # Limit thread count here
        #VECs[i].setenv('OMP_NUM_THREADS', str(NTHREADS))
        with VECs[i]:
            #import mkl
            #mkl.set_num_threads(int(NTHREADS))
            import numpy as np

        # Populate VEC queue
        VEC_q.put(i)
    
    # Reserve last context for single threaded stuff
    # Unlimited threads, don't setenv
    with VECs[NGROUPS]:
        import numpy as np # TODO Make sure this gets all threads
    main_VEC = VECs[NGROUPS]

    for i in range(WARMUP + ITERS):
        with main_VEC:
            # Original matrix
            np.random.seed(i)
            A = np.random.rand(NROWS, NCOLS)

        # Multithreaded blocked version with VECs
        start = time.time()
        Q, R = tsqr_blocked(A)
        end = time.time()

        if (i >= WARMUP):
            print(end - start)

        if CHECK_RESULT:
            with main_VEC:
                print(check_result(A, Q, R))

    print('\n%**********************************************************************************************%\n')
