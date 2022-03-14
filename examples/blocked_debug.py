"""
A naive implementation of blocked Cholesky using Numba kernels on CPUs.
"""

import numpy as np
from numba import jit, void, float64
import math
import time

from parla import Parla, get_all_devices

from parla.cuda import gpu
from parla.cpu import cpu

from parla.function_decorators import specialized
from parla.tasks import *
from parla.ldevice import LDeviceGridBlocked

import cupy as cp
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import _util

from parla.parray import asarray_batch

from scipy import linalg
import sys

block_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32*5
n = block_size*int(sys.argv[2]) if len(sys.argv) > 2 else block_size*16

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

def flatten(t):
    return [item for sublist in t for item in sublist]

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

    print("Starting..", flush=True)
    print("Initial Array", a, flush=True)
    block_size = a[0][0].array.shape[0]
    # Define task spaces
    gemm1 = TaskSpace("gemm1")        # Inter-block GEMM
    subcholesky = TaskSpace("subcholesky")  # Cholesky on block
    gemm2 = TaskSpace("gemm2")        # Inter-block GEMM
    solve = TaskSpace("solve")        # Triangular solve

    for j in range(len(a)):
        for k in range(j):
            # Inter-block GEMM
            mem = 4* block_size**2
            @spawn(gemm1[j, k], [solve[j, k]], input=[a[j][k]], inout=[a[j][j]], placement=loc, memory=mem)
            def t1():
                print(f"+SYRK: ({j}, {k}) - Requires rw({j},{j})  r({j}, {k})", flush=True)
                out = a[j][j].array
                rhs = a[j][k].array
                if True or (out is None) or (rhs is None):
                    print("gemm1", j, k, out, rhs, flush=True)
                out[:] = update(rhs, rhs, out)
                if (out is None):
                    print("gemm1 problem on exit", flush=True)

                stream = cp.cuda.get_current_stream()
                stream.synchronize()
                a[j][j].update(out)
                stream.synchronize()
                print("SYRK A[j, j]", a[j][j].array, flush=True)
                print(f"-SYRK: ({j}, {k}) - Requires rw({j},{j})  r({j}, {k})", flush=True)

        # Cholesky on block
        mem = 2* block_size**2
        @spawn(subcholesky[j], [gemm1[j, 0:j]], inout=[a[j][j]], placement=loc, memory=mem)
        def t2():
            print(f"+POTRF: ({j}) - Requires rw({j},{j})", flush=True)
            dblock = a[j][j].array

            if True or (dblock is None):
                print("potrf", j, dblock, flush=True)

            dblock = cholesky(dblock)

            if dblock is None:
                print("POTRF problem on exit", flush=True)

            stream = cp.cuda.get_current_stream()
            stream.synchronize()
            a[j][j].update(dblock)
            stream.synchronize()
            print("POTRF A[j, j]", a[j][j].array, flush=True)
            print(f"-POTRF: ({j}) - Requires rw({j},{j})", flush=True)

        for i in range(j+1, len(a)):
            for k in range(j):
                # Inter-block GEMM
                mem = 5*block_size**2
                @spawn(gemm2[i, j, k], [solve[j, k], solve[i, k]], inout=[a[i][j]], input=[a[i][k], a[j][k]], placement=loc, memory=mem)
                def t3():
                    print(f"+GEMM: ({i}, {j}, {k}) - Requires rw({i},{j}), r({i}, {k}), r({j}, {k})", flush=True)
                    out = a[i][j].array
                    rhs1 = a[i][k].array
                    rhs2 = a[j][k].array
                    if True or (out is None) or (rhs1 is None) or (rhs2 is None):
                        print("gemm2", i, j, k, out, rhs1, rhs2, flush=True)
                    out = update(rhs1, rhs2, out)
                    if out is None:
                        print("gemm2 problem on exit", flush=True)
                    stream = cp.cuda.get_current_stream()
                    stream.synchronize()
                    a[i][j].update(out)
                    stream.synchronize()
                    print("GEMM A[i, j]", a[i][j].array, flush=True)
                    print(f"-GEMM: ({i}, {j}, {k}) - Requires rw({i},{j}), r({i}, {k}), r({j}, {k})", flush=True)

            # Triangular solve
            mem = 3*block_size**2
            @spawn(solve[i, j], [gemm2[i, j, 0:j], subcholesky[j]], inout=[a[i][j]], input=[a[j][j]], placement=loc, memory=mem)
            def t4():
                print(f"+TRSM: ({i}, {j}) - Requires rw({i},{j}), r({j}, {j})", flush=True)
                factor = a[j][j].array
                panel = a[i][j].array
                if True or (factor is None) or (panel is None):
                    print("trsm", i, j, factor, panel, flush=True)
                out = ltriang_solve(factor, panel)
                if out is None:
                    print("trsm problem on exit", flush=True)
                stream = cp.cuda.get_current_stream()
                stream.synchronize()
                a[i][j].update(out)
                stream.synchronize()
                print("TRSM A[i, j]", a[i][j].array, flush=True)
                print(f"-TRSM: ({i}, {j}) - Requires rw({i},{j}), r({j}, {j})", flush=True)


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

        ap = a1.reshape(n//block_size, block_size, n//block_size, block_size).swapaxes(1, 2)

        ap_parray_list = list()
        for i in range(n//block_size):
            ap_parray_list.append(list())
            for j in range(n//block_size):
                ap_parray_list[i].append(ap[i][j])

        ap_parray = asarray_batch(ap_parray_list)
        print(ap_parray)


        start = time.perf_counter()

        # Call Parla Cholesky result and wait for completion
        await cholesky_blocked_inplace(ap_parray)

        #print(ap)
        end = time.perf_counter()

        ts = TaskSpace("CopyBack")
        @spawn(taskid=ts[0], placement=cpu, inout=flatten(ap_parray))
        def copy_back():
            for i in range(n//block_size):
                for j in range(n//block_size):
                    print()
                    print(ap_parray[i][j].array)
                    ap[i,j][:] = ap_parray[i][j].array

        await ts

        print(end - start, "seconds")

        #print("Truth", linalg.cholesky(a).T)
        print("a1", a1)
        print("ap", ap)
        #Check result
        computed_L = np.tril(a1)
        #print("Soln", computed_L)
        error = np.max(np.absolute(a - computed_L @ computed_L.T))
        print("Error", error)
        #assert(error < 1E-8)

if __name__ == '__main__':
    with Parla():
        main()
