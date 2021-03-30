"""
A naive implementation of blocked Cholesky using Numba kernels on CPUs.
"""

import numpy as np
#import cupy
from numba import jit, void, float64
import math
import time

from parla import Parla, get_all_devices
from parla.array import copy, clone_here

from parla.cuda import gpu
from parla.cpu import cpu

from parla.function_decorators import specialized
from parla.tasks import *

import cupy as cp
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import _util

from scipy import linalg
import sys

block_size = int(sys.argv[1])
n = block_size*int(sys.argv[2])

loc = gpu

@specialized
@jit(float64[:,:](float64[:,:]), nopython=True, nogil=True)
def cholesky(a):
    """
    Naive version of dpotrf. Write results into lower triangle of the input array.
    """
    if a.shape[0] != a.shape[1]:
        raise ValueError("A square array is required.")
    for j in range(a.shape[0]):
        a[j,j] = math.sqrt(a[j,j] - (a[j,:j] * a[j,:j]).sum())
        for i in range(j+1, a.shape[0]):
            a[i,j] -= (a[i,:j] * a[j,:j]).sum()
            a[i,j] /= a[j,j]
    return a


@cholesky.variant(gpu)
def choleksy_gpu(a):
    a = cp.linalg.cholesky(a)
    if cp.any(cp.isnan(a)):
      raise np.linalg.LinAlgError
    return a

@specialized
@jit(float64[:,:](float64[:,:], float64[:,:]), nopython=True, nogil=True)
def ltriang_solve(a, b):
    """
    This is a naive version of dtrsm. The result is written over the input array `b`.
    """
    b = b.T
    if a.shape[0] != b.shape[0]:
        raise ValueError("Input array shapes are not compatible.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Array for back substitution is not square.")
    #For the implementation here, just assume lower triangular.
    for i in range(a.shape[0]):
        b[i] /= a[i,i]
        b[i+1:] -= a[i+1:,i:i+1] * b[i:i+1]
    return b.T

#comments would repack the data to column - major
def cupy_trsm_wrapper(a, b):
    cublas_handle = device.get_cublas_handle()
    trsm = cublas.dtrsm
    uplo = cublas.CUBLAS_FILL_MODE_LOWER
    a = cp.array(a, dtype=np.float64, order='F')
    b = cp.array(b, dtype=np.float64, order='F')
    trans = cublas.CUBLAS_OP_T
    side = cublas.CUBLAS_SIDE_RIGHT

    #trans = cublas.CUBLAS_OP_T
    #side = cublas.CUBLAS_SIDE_LEFT

    diag = cublas.CUBLAS_DIAG_NON_UNIT
    m, n = (b.side, 1) if b.ndim == 1 else b.shape
    trsm(cublas_handle, side, uplo, trans, diag, m, n, 1.0, a.data.ptr, m, b.data.ptr, m)
    return b

@ltriang_solve.variant(gpu)
def ltriang_solve_gpu(a, b):
    b = cupy_trsm_wrapper(a, b)
    return b

def update_kernel(a, b, c):
    c -= a @ b.T
    return c

@specialized
def update(a, b, c):
    c = update_kernel(a, b, c)
    return c

@update.variant(gpu)
def update_gpu(a, b, c):
    c = update_kernel(a, b, c)
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

    #Define task spaces
    gemm1 = TaskSpace("gemm1")        # Inter-block GEMM
    subcholesky = TaskSpace("subcholesky")  # Cholesky on block
    gemm2 = TaskSpace("gemm2")        # Inter-block GEMM
    solve = TaskSpace("solve")        # Triangular solve

    for j in range(a.shape[0]):
        for k in range(j):
            #Inter - block GEMM
            @spawn(gemm1[j, k], [solve[j, k]], placement=loc)
            def t1():
                out = clone_here(a[j,j])  # Move data to the current device
                rhs = clone_here(a[j,k])
                out = update(rhs, rhs, out)
                copy(a[j,j], out)  # Move the result to the global array

        #Cholesky on block
        @spawn(subcholesky[j], [gemm1[j, 0:j]], placement=loc)
        def t2():
            dblock = clone_here(a[j, j])
            dblock = cholesky(dblock)
            copy(a[j, j], dblock)

        for i in range(j+1, a.shape[0]):
            for k in range(j):
                #Inter - block GEMM
                @spawn(gemm2[i, j, k], [solve[j, k], solve[i, k]], placement=loc)
                def t3():
                    out = clone_here(a[i,j])  # Move data to the current device
                    rhs1 = clone_here(a[i,k])
                    rhs2 = clone_here(a[j,k])
                    out = update(rhs1, rhs2, out)
                    copy(a[i,j], out)  # Move the result to the global array

            #Triangular solve
            @spawn(solve[i, j], [gemm2[i, j, 0:j], subcholesky[j]], placement=loc)
            def t4():
                factor = clone_here(a[j, j])
                panel = clone_here(a[i, j])
                out = ltriang_solve(factor, panel)
                copy(a[i, j], out)

    return subcholesky[a.shape[0]-1]

def main():
    @spawn(placement=cpu)
    async def test_blocked_cholesky():
        assert not n % block_size

        np.random.seed(10)
        #Construct input data
        a = np.random.rand(n, n)
        a = a @ a.T

        #Copy and layout input
        a1 = a.copy()
        ap = a1.reshape(n // block_size, block_size, n // block_size, block_size).swapaxes(1,2)
        start = time.perf_counter()

        #Call Parla Cholesky result and wait for completion
        await cholesky_blocked_inplace(ap)

        end = time.perf_counter()
        print(end - start, "seconds")

        print("Truth", linalg.cholesky(a).T)

        #Check result
        computed_L = np.tril(a1)
        print("Soln", computed_L)
        error = np.max(np.absolute(a-computed_L @ computed_L.T))
        print("Error", error)
        assert(error < 1E-8)

if __name__ == '__main__':
    with Parla():
        main()
