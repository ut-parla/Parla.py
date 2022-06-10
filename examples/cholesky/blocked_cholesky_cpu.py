"""
A naive implementation of blocked Cholesky using Numba kernels on CPUs.
"""
import os
import mkl
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=2000)
parser.add_argument('-nblocks', type=int, default=10)
parser.add_argument('-trials', type=int, default=1)
parser.add_argument('-matrix', default=None)
parser.add_argument('-fixed', default=0, type=int)
parser.add_argument('-t', type=int, default=1)
parser.add_argument('-workers', type=int, default=1)
args = parser.parse_args()

load = 1.0/args.workers
mkl.set_num_threads(args.t)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.t)
os.environ["OMP_NUM_THREADS"] = str(args.t)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.t)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.t)

block_size = args.b
fixed = args.fixed

if args.matrix is None:
    n = block_size * args.nblocks

num_tests = args.trials

save_file = True
check_nan = True
check_error = True

import numpy as np
from scipy import linalg

import os
import time

from parla import Parla, get_all_devices
from parla.array import copy, clone_here
from parla.cpu import cpu

from parla.function_decorators import specialized
from parla.tasks import spawn, TaskSpace, CompletedTaskSpace

try:
    from parla.cuda import get_memory_log, summarize_memory, log_memory, clean_memory
except (ImportError, AttributeError):
    def get_memory_log():
        pass
    def summarize_memory():
        pass
    def log_memory():
        pass
    def clean_memory():
        pass

from dask.array.utils import array_safe, meta_from_array, solve_triangular_safe

from numba import jit, void, float64
import math

#This triangular solve only supports one side
#import cupyx.scipy.linalg as cpx

loc = cpu

<<<<<<< HEAD

=======
>>>>>>> main
def numpy_trsm_wrapper(a, b):
    a = np.array(a, order='F', dtype=np.float64)
    b = np.array(b, order='F', dtype=np.float64)
    b = linalg.blas.dtrsm(1.0, a, b, trans_a=1, lower=1, side=1)
    return b

def cholesky_inplace(a):
    a = linalg.cholesky(a, lower=True)
    return a

def ltriang_solve(a, b):
    b = solve_triangular_safe(a, b.T, lower=True)
    #b = numpy_trsm_wrapper(a, b)
    return b.T

def update_kernel(a, b, c):
    c -= a @ b.T
    return c

def update(a, b, c):
    c = update_kernel(a, b, c)
    #c = linalg.blas.dgemm(-1.0, a, b, c=c, beta=1.0, overwrite_c=True, trans_a=False, trans_b=True)
    return c

def cholesky_blocked_inplace(a):
    """
    This is a less naive version of dpotrf with one level of blocking.
    Blocks are currently assumed to evenly divide the axes lengths.
    The input array 4 dimensional. The first and second index select
    the block (row first, then column). The third and fourth index
    select the entry within the given block.
    """

    # Define task spaces
    syrk = TaskSpace("syrk")        # Inter-block GEMM
    subcholesky = TaskSpace("subcholesky")  # Cholesky on block
    gemm = TaskSpace("gemm")        # Inter-block GEMM
    solve = TaskSpace("solve")        # Triangular solve
    zerofy = TaskSpace("zerofy")

    for j in range(len(a)):
        for k in range(j):
            # Inter-block GEMM
            @spawn(syrk[j, k], [solve[j, k], syrk[j, 0:k]], placement=loc, vcus=load)
            #@spawn(syrk[j, k], [solve[j, k]])
            def t0():
                out = clone_here(a[j][j])  # Move data to the current device
                rhs = clone_here(a[j][k])

                out = update(rhs, rhs, out)

                copy(a[j][j], out)  # Move the result to the global array

        # Cholesky on block

        @spawn(subcholesky[j], [syrk[j, 0:j]], placement=loc, vcus=load)
        def t2():
            dblock = clone_here(a[j][j])
            dblock = cholesky_inplace(dblock)
            copy(a[j][j], dblock)

        for i in range(j+1, len(a)):
            for k in range(j):
                # Inter-block GEMM
                @spawn(gemm[i, j, k], [solve[j, k], solve[i, k], gemm[i, j, 0:k]], placement=loc, vcus=load)
                def t3():
                    out = clone_here(a[i][j])  # Move data to the current device
                    rhs1 = clone_here(a[i][k])
                    rhs2 = clone_here(a[j][k])

                    out = update(rhs1, rhs2, out)

                    copy(a[i][j], out)  # Move the result to the global array

            # Triangular solve
            @spawn(solve[i, j], [gemm[i, j, 0:j], subcholesky[j]], placement=loc, vcus=load)
            def t4():
                factor = clone_here(a[j][j])
                panel = clone_here(a[i][j])
                panel = ltriang_solve(factor, panel)
                copy(a[i][j], panel)
<<<<<<< HEAD

    @spawn(zerofy[0], [subcholesky[len(a) - 1]], placement=loc)
    def t5():
        for i in range(len(a)):
            for j in range(len(a)):
                if j < i:
                    a[i][j] = 0

    return zerofy[0]
    #return subcholesky[len(a)-1]
=======
    return subcholesky[len(a)-1]
>>>>>>> main

def main():
    @spawn(placement=cpu)
    async def test_blocked_cholesky():
        global n

        if args.matrix is None:
            print("Generating matrix of size: ", n)
            # Construct input data
            a = np.random.rand(n, n)
            a = a @ a.T

            if save_file:
                np.save(f"chol_{n}", a)
        else:
            print("Loading matrix from file: ", args.matrix)
            a = np.load(args.matrix)
            print("Loaded matrix from file. Shape=", a.shape)
            n = a.shape[0]

        # Copy and layout input
        print("Blocksize: ", block_size)
        assert not n % block_size
        a1 = a.copy()
        #a_temp = a1.reshape(n//block_size, block_size, n//block_size, block_size).swapaxes(1, 2)

        for k in range(num_tests):
            ap = a1.copy()
<<<<<<< HEAD

=======
>>>>>>> main
            ap_list = list()
            for i in range(n//block_size):
                ap_list.append(list())
                for j in range(n//block_size):
                    ap_list[i].append(np.copy(a1[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size], order='F'))

            print("Starting Cholesky")
            print("------------")

            start = time.perf_counter()
            # Call Parla Cholesky result and wait for completion
            await cholesky_blocked_inplace(ap_list)
            end = time.perf_counter()

<<<<<<< HEAD
            print(f"Trial {k}:", end - start, "seconds")
=======
>>>>>>> main
            summarize_memory()
            clean_memory()
            print("--------")

            ts = TaskSpace("CopyBack")
            @spawn(taskid=ts[0], placement=cpu)
            def copy_back():
                for i in range(n//block_size):
                    for j in range(n//block_size):
                        ap[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size] = ap_list[i][j]

            await ts

<<<<<<< HEAD
=======
            zerofy_start = time.perf_counter()
            computed_L = np.tril(ap)
            zerofy_end = time.perf_counter()

            print(f"Trial {k}:", (end - start) + (zerofy_end - zerofy_start), "seconds")

>>>>>>> main
            # Check result
            print("Is NAN: ", np.isnan(np.sum(ap)))

            if check_error:
<<<<<<< HEAD
                computed_L = ap
=======
>>>>>>> main
                error = np.max(np.absolute(a - computed_L @ computed_L.T))
                print("Error", error)

if __name__ == '__main__':
    np.random.seed(10)
    random.seed(10)

    with Parla():
        main()
