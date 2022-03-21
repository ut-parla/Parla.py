"""
A naive implementation of blocked Cholesky using Numba kernels on CPUs.
"""

import numpy as np
from numba import jit, void, float64
import math
import time

from parla import Parla, get_all_devices

from parla.cuda import gpu

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

from parla.cpu import cpu

from parla.function_decorators import specialized
from parla.tasks import *

#from parla.task_runtime import get_current_devices
from parla.ldevice import LDeviceGridBlocked
from parla.array import clone_here

import cupy as cp
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import _util

from scipy import linalg
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=32*5)
parser.add_argument('-nblocks', type=int, default=16)
parser.add_argument('-trials', type=int, default=1)
args = parser.parse_args()

block_size = args.b
n = block_size*args.nblocks
num_tests = args.trials

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
        a[j, j] = math.sqrt(a[j, j] - (a[j, :j] * a[j, :j]).sum())
        for i in range(j+1, a.shape[0]):
            a[i, j] -= (a[i, :j] * a[j, :j]).sum()
            a[i, j] /= a[j, j]
    return a


@cholesky.variant(gpu)
def cholesky_gpu(a):
    a = cp.linalg.cholesky(a)
    #if cp.any(cp.isnan(a)):
    #    print(a, flush=True)
    #    raise np.linalg.LinAlgError
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
    # For the implementation here, just assume lower triangular.
    for i in range(a.shape[0]):
        b[i] /= a[i, i]
        b[i+1:] -= a[i+1:, i:i+1] * b[i:i+1]
    return b.T

# comments would repack the data to column - major
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
    # TODO (bozhi): these should be guaranteed by the partitioner
    if len(a) * a[0][0].shape[0] != len(a[0]) * a[0][0].shape[1]:
        raise ValueError("A square matrix is required.")
    if len(a) != len(a[0]):
        raise ValueError("Non-square blocks are not supported.")

    # Define task spaces
    gemm1 = TaskSpace("gemm1")        # Inter-block GEMM
    subcholesky = TaskSpace("subcholesky")  # Cholesky on block
    gemm2 = TaskSpace("gemm2")        # Inter-block GEMM
    solve = TaskSpace("solve")        # Triangular solve

    for j in range(len(a)):
        for k in range(j):
            # Inter-block GEMM
            @spawn(gemm1[j, k], [solve[j, k], gemm1[j, 0:k]], placement=loc)
            def t1():
                #print("GEMM1", (j, j), (j, k), flush=True)
                out = clone_here(a[j][j])
                rhs = clone_here(a[j][k])

                #stream = cp.cuda.get_current_stream()
                #stream.synchronize()

                out = update(rhs, rhs, out)

                #stream = cp.cuda.get_current_stream()
                #stream.synchronize()
                log_memory()
                a[j][j] = out
                #stream.synchronize()

        # Cholesky on block
        @spawn(subcholesky[j], [gemm1[j, 0:j]], placement=loc)
        def t2():
            dblock = clone_here(a[j][j])
            #stream = cp.cuda.get_current_stream()
            #stream.synchronize()
            #print(j, dblock, flush=True)
            dblock = cholesky(dblock)
            #stream = cp.cuda.get_current_stream()
            #stream.synchronize()
            log_memory()
            a[j][j] = dblock
            #stream.synchronize()

        for i in range(j+1, len(a)):
            for k in range(j):
                # Inter-block GEMM
                @spawn(gemm2[i, j, k], [solve[j, k], solve[i, k], gemm2[i, j, 0:k]], placement=loc)
                def t3():

                    #print("GEMM2", (i, j), (i, k), (j, k) , flush=True)
                    out = clone_here(a[i][j])
                    rhs1 = clone_here(a[i][k])
                    rhs2 = clone_here(a[j][k])
                    #stream = cp.cuda.get_current_stream()
                    #stream.synchronize()

                    out = update(rhs1, rhs2, out)
                    #stream = cp.cuda.get_current_stream()
                    #stream.synchronize()
                    log_memory()
                    a[i][j] = out
                    #stream.synchronize()


            # Triangular solve
            @spawn(solve[i, j], [gemm2[i, j, 0:j], subcholesky[j]], placement=loc)
            def t4():

                factor = clone_here(a[j][j])
                panel = clone_here(a[i][j])

                #stream = cp.cuda.get_current_stream()
                #stream.synchronize()
                out = ltriang_solve(factor, panel)
                #stream = cp.cuda.get_current_stream()
                #stream.synchronize()
                log_memory()
                a[i][j] = out
                #stream.synchronize()

    return subcholesky[len(a)-1]

def main():
    @spawn(placement=cpu)
    async def test_blocked_cholesky():
        assert not n % block_size

        np.random.seed(10)
        # Construct input data
        a = np.random.rand(n, n)
        a = a @ a.T

        # Copy and layout input

        a1 = a.copy()
        a_temp = a1.reshape(n//block_size, block_size, n//block_size, block_size).swapaxes(1, 2)

        n_gpus = cp.cuda.runtime.getDeviceCount()

        for k in range(num_tests):
            ap = a_temp.copy()


            ap_list = list()
            for i in range(n//block_size):
                ap_list.append(list())
                for j in range(n//block_size):
                    with cp.cuda.Device(i%n_gpus):
                        ap_list[i].append(cp.asarray(ap[i][j]))

            start = time.perf_counter()
            # Call Parla Cholesky result and wait for completion
            await cholesky_blocked_inplace(ap_list)
            end = time.perf_counter()

            print(f"Trial {k}:", end - start, "seconds")
            summarize_memory()
            clean_memory()
            print("--------")


            ts = TaskSpace("CopyBack")
            @spawn(taskid=ts[0], placement=cpu)
            def copy_back():
                for i in range(n//block_size):
                    for j in range(n//block_size):
                        a1[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size] = ap_list[i][j].get()

            await ts



            # Check result
            computed_L = np.tril(a1)
            error = np.max(np.absolute(a - computed_L @ computed_L.T))
            print("Error", error)

if __name__ == '__main__':
    with Parla():
        main()
