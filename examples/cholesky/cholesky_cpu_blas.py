"""
A naive implementation of blocked Cholesky using Numba kernels on CPUs.
"""

import numpy as np
from scipy import linalg
import cupy as cp

import os
import time

from parla import Parla, get_all_devices
from parla.array import copy, clone_here
from parla.cuda import gpu
from parla.cpu import cpu

from parla.function_decorators import specialized
from parla.tasks import spawn, TaskSpace, CompletedTaskSpace

from dask.array.utils import array_safe, meta_from_array, solve_triangular_safe

from cupy.cuda import cublas
from cupy.cuda import device
#from cupy.linalg import _util

from numba import jit, void, float64
import math

#This triangular solve only supports one side
#import cupyx.scipy.linalg as cpx

loc = cpu

os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

#Contains no error checking
def cupy_trsm_wrapper(a, b):
    cublas_handle = device.get_cublas_handle()
    a = cp.array(a, dtype=np.float64, order='F')
    b = cp.array(b, dtype=np.float64, order='F')
    trsm = cublas.dtrsm
    uplo = cublas.CUBLAS_FILL_MODE_LOWER
    trans = cublas.CUBLAS_OP_T
    side = cublas.CUBLAS_SIDE_RIGHT
    diag = cublas.CUBLAS_DIAG_NON_UNIT
    m, n = (b.side, 1) if b.ndim == 1 else b.shape
    trsm(cublas_handle, side, uplo, trans, diag, m, n, 1.0, a.data.ptr, m, b.data.ptr, m)
    return b

def numpy_trsm_wrapper(a, b):
    a = np.array(a, order='F', dtype=np.float64)
    b = np.array(b, order='F', dtype=np.float64)
    b = linalg.blas.dtrsm(1.0, a, b, trans_a=1, lower=1, side=1)
    return b

@specialized
def cholesky_inplace(a):
    a = linalg.cholesky(a, lower=True)
    return a

@cholesky_inplace.variant(gpu)
def cholesky_inplace_gpu(a):
    #only supports lower triangular
    a = cp.linalg.cholesky(a)
    return a


@specialized
def ltriang_solve(a, b):
    b = solve_triangular_safe(a, b.T, lower=True)
    #b = numpy_trsm_wrapper(a, b)
    return b.T

@ltriang_solve.variant(gpu)
def ltriang_solve_gpu(a, b):
    b = cupy_trsm_wrapper(a, b)
    #cpx.solve_triangular(a, b, trans='T', overwrite_b=True, lower=True)
    return b


def update_kernel(a, b, c):
    c -= a @ b.T
    return c

@specialized
def update(a, b, c):
    c = update_kernel(a, b, c)
    #c = linalg.blas.dgemm(-1.0, a, b, c=c, beta=1.0, overwrite_c=True, trans_a=False, trans_b=True)     
    return c

@update.variant(gpu)
def update_gpu(a, b, c):
    c = update_kernel(a, b, c)
    #cp.cuda.cublas.dgemm('N', 'T', a, b, out=C, alpha=-1.0, beta=1)
    return c

def cholesky_blocked_inplace(a):
    """
    This is a less naive version of dpotrf with one level of blocking.
    Blocks are currently assumed to evenly divide the axes lengths.
    The input array 4 dimensional. The first and second index select
    the block (row first, then column). The third and fourth index
    select the entry within the given block.
    """
    if a.shape[0] * a.shape[2] != a.shape[1] * a.shape[3]:
        raise ValueError("A square matrix is required.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Non-square blocks are not supported.")

    # Define task spaces
    syrk = TaskSpace("syrk")        # Inter-block GEMM
    subcholesky = TaskSpace("subcholesky")  # Cholesky on block
    gemm = TaskSpace("gemm")        # Inter-block GEMM
    solve = TaskSpace("solve")        # Triangular solve
    zerofy = TaskSpace("zerofy")

    for j in range(a.shape[0]):
        for k in range(j):
            # Inter-block GEMM
            @spawn(syrk[j, k], [solve[j, k]], placement=loc)
            #@spawn(syrk[j, k], [solve[j, k]])
            def t1():
                out = clone_here(a[j,j])  # Move data to the current device
                rhs = clone_here(a[j,k])

                out = update(rhs, rhs, out)

                copy(a[j,j], out)  # Move the result to the global array

        # Cholesky on block

        @spawn(subcholesky[j], [syrk[j, 0:j]], placement=loc)
        def t2():
            dblock = clone_here(a[j, j])
            dblock = cholesky_inplace(dblock)
            copy(a[j, j], dblock)

        for i in range(j+1, a.shape[0]):
            for k in range(j):
                # Inter-block GEMM
                @spawn(gemm[i, j, k], [solve[j, k], solve[i, k]], placement=loc)
                def t3():
                    out = clone_here(a[i,j])  # Move data to the current device
                    rhs1 = clone_here(a[i,k])
                    rhs2 = clone_here(a[j,k])

                    out = update(rhs1, rhs2, out)

                    copy(a[i,j], out)  # Move the result to the global array

            # Triangular solve
            @spawn(solve[i, j], [gemm[i, j, 0:j], subcholesky[j]], placement=loc)
            def t4():
                factor = clone_here(a[j, j])
                panel = clone_here(a[i, j])
                panel = ltriang_solve(factor, panel)
                copy(a[i, j], panel)

    """
    @spawn(zerofy, [subcholesky[a.shape[0]-1]], placement=loc)
    def zero():
        target = clone_here(a)
        print(">> before:", target)
        computed_L = np.tril(target)
        print(">> computed:", computed_L)
        copy(a, computed_L)
    """
        
    return subcholesky[a.shape[0]-1]

def main():
    @spawn(placement=cpu)
    async def test_blocked_cholesky():

        np.random.seed(10)

        # Configure environment
        block_size = 2000
        n = 20000
        assert not n % block_size

        # Construct input data
        a = np.random.rand(n, n)
        a = a @ a.T

        # Copy and layout input
        a1 = a.copy()
        ap = a1.reshape(n // block_size, block_size, n // block_size, block_size).swapaxes(1,2)
        start = time.perf_counter()

        # Call Parla Cholesky result and wait for completion
        await cholesky_blocked_inplace(ap)
        #print(ap)

        end = time.perf_counter()
        print(end - start, "seconds")
        #print("Truth", linalg.cholesky(a))
        # Check result
        computed_L = np.tril(a1)
        #print("Soln", computed_L)
        print(np.max(np.absolute(a - computed_L @ computed_L.T)))
        assert(np.max(np.absolute(a - computed_L @ computed_L.T)) < 1E-6)

if __name__ == '__main__':
    with Parla():
        main()
