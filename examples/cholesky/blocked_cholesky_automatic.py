"""
A naive implementation of blocked Cholesky using Numba kernels on CPUs.
"""

import argparse
import os

parser = argparse.ArgumentParser()
#Blocksize
parser.add_argument('-b', type=int, default=2000)
#How many blocks
parser.add_argument('-nblocks', type=int, default=14)
#How many trials to run
parser.add_argument('-trials', type=int, default=1)
#What matrix file (.npy) to load
parser.add_argument('-matrix', default=None)
#Are the placements fixed by the user or determined by the scheduler?
parser.add_argument('-fixed', default=0, type=int)
#How many GPUs to run on?
parser.add_argument('-ngpus', default=4, type=int)
args = parser.parse_args()

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    print("CUDA_VISIBLE_DEVICES is not set. Assuming 0-3")
    cuda_visible_devices = list(range(4))
else:
    cuda_visible_devices = cuda_visible_devices.strip().split(',')
    cuda_visible_devices = list(map(int, cuda_visible_devices))

gpus = cuda_visible_devices[:args.ngpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

import random
import numpy as np
from numba import jit, float64
import math
import time

from parla import Parla

from parla.cuda import gpu
from parla.cpu import cpu

from parla.function_decorators import specialized
from parla.tasks import *

import cupy as cp
from cupy.cuda import cublas
from cupy.cuda import device

from parla.parray import asarray_batch


ngpus = cp.cuda.runtime.getDeviceCount()
#Make sure that the enviornment configuration is correct
assert(ngpus == args.ngpus)

block_size = args.b
fixed = args.fixed

if args.matrix is None:
    n = block_size*args.nblocks

num_tests = args.trials

loc = gpu

save_file = True
check_nan = False
check_error = False
time_zeros = False   #Set to true if comparing with Dask.

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


from scipy import linalg
import sys

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
    #  raise np.linalg.LinAlgError
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
    a = cp.asarray(a, dtype=np.float64, order='F')
    b = cp.asarray(b, dtype=np.float64, order='F')
    trans = cublas.CUBLAS_OP_T
    side = cublas.CUBLAS_SIDE_RIGHT

    #trans = cublas.CUBLAS_OP_T
    #side = cublas.CUBLAS_SIDE_LEFT

    diag = cublas.CUBLAS_DIAG_NON_UNIT
    m, n = (b.side, 1) if b.ndim == 1 else b.shape
    alpha = np.array(1, dtype=a.dtype)
    # Cupy >= 9 requires pointers even for coefficients.
    # https://github.com/cupy/cupy/issues/7011
    trsm(cublas_handle, side, uplo, trans, diag, m, n, alpha.ctypes.data, a.data.ptr, m, b.data.ptr, m)
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

def cholesky_blocked_inplace(a, block_size):
    """
    This is a less naive version of dpotrf with one level of blocking.
    Blocks are currently assumed to evenly divide the axes lengths.
    The input array 4 dimensional. The first and second index select
    the block (row first, then column). The third and fourth index
    select the entry within the given block.
    """
    # TODO (bozhi): these should be guaranteed by the partitioner
    #if len(a) * a[0][0].shape[0] != len(a[0]) * a[0][0].shape[1]:
    #    raise ValueError("A square matrix is required.")
    #if len(a) != len(a[0]):
    #    raise ValueError("Non-square blocks are not supported.")

    #print("Starting..", flush=True)
    #print("Initial Array", a, flush=True)
    #block_size = a[0][0].array.shape[0]
    # Define task spaces
    gemm1 = TaskSpace("gemm1")        # Inter-block GEMM
    subcholesky = TaskSpace("subcholesky")  # Cholesky on block
    gemm2 = TaskSpace("gemm2")        # Inter-block GEMM
    solve = TaskSpace("solve")        # Triangular solve

    for j in range(len(a)):
        for k in range(j):
            # Inter-block GEMM
            mem = 8*block_size**2


            loc_syrk = gpu
            if fixed:
                loc_syrk = gpu(j%ngpus)

            @spawn(gemm1[j, k], [solve[j, k], gemm1[j, 0:k]], input=[a[j][k]], inout=[a[j][j]], placement=loc_syrk, memory=mem)
            def t1():
                #print(f"+SYRK: ({j}, {k}) - Requires rw({j},{j})  r({j}, {k})", flush=True)
                out = a[j][j].array
                rhs = a[j][k].array
                out = update(rhs, rhs, out)

                stream = cp.cuda.get_current_stream()
                stream.synchronize()
                a[j][j].update(out)
                stream.synchronize()
                #print(f"-SYRK: ({j}, {k}) - Requires rw({j},{j})  r({j}, {k})", flush=True)

        # Cholesky on block
        mem = 8*block_size**2


        loc_potrf = gpu
        if fixed:
            loc_potrf = gpu(j%ngpus)

        @spawn(subcholesky[j], [gemm1[j, 0:j]], inout=[a[j][j]], placement=loc_potrf, memory=mem)
        def t2():
            #print(f"+POTRF: ({j}) - Requires rw({j},{j})", flush=True)
            dblock = a[j][j].array

            log_memory()
            dblock = cholesky(dblock)

            stream = cp.cuda.get_current_stream()
            stream.synchronize()
            a[j][j].update(dblock)
            stream.synchronize()
            #print(f"-POTRF: ({j}) - Requires rw({j},{j})", flush=True)
        for i in range(j+1, len(a)):
            for k in range(j):
                # Inter-block GEMM
                mem = 8*block_size**2

                loc_gemm = gpu
                if fixed:
                    loc_gemm = gpu(i%ngpus)

                @spawn(gemm2[i, j, k], [solve[j, k], solve[i, k], gemm2[i, j, 0:k]], inout=[a[i][j]], input=[a[i][k], a[j][k]], placement=loc_gemm, memory=mem)
                def t3():
                    #print(f"+GEMM: ({i}, {j}, {k}) - Requires rw({i},{j}), r({i}, {k}), r({j}, {k})", flush=True)
                    out = a[i][j].array
                    rhs1 = a[i][k].array
                    rhs2 = a[j][k].array

                    stream = cp.cuda.get_current_stream()

                    log_memory()
                    out = update(rhs1, rhs2, out)
                    stream.synchronize()
                    a[i][j].update(out)
                    stream.synchronize()
                    #print(f"-GEMM: ({i}, {j}, {k}) - Requires rw({i},{j}), r({i}, {k}), r({j}, {k})", flush=True)

            # Triangular solve
            mem = 8*2*block_size**2


            loc_trsm = gpu
            if fixed:
                loc_trsm = gpu(i%ngpus)

            @spawn(solve[i, j], [gemm2[i, j, 0:j], subcholesky[j]], inout=[a[i][j]], input=[a[j][j]], placement=loc_trsm, memory=mem)
            def t4():
                #print(f"+TRSM: ({i}, {j}) - Requires rw({i},{j}), r({j}, {j})", flush=True)
                factor = a[j][j].array
                panel = a[i][j].array

                log_memory()
                out = ltriang_solve(factor, panel)
                stream = cp.cuda.get_current_stream()
                stream.synchronize()
                a[i][j].update(out)
                stream.synchronize()
                #print(f"-TRSM: ({i}, {j}) - Requires rw({i},{j}), r({j}, {j})", flush=True)

    return subcholesky[len(a) - 1]

def main():
    @spawn(placement=cpu)
    async def test_blocked_cholesky():
        global n

        if args.matrix is None:
            print("Generating matrix of size: ", n)
            np.random.seed(10)
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

        n_gpus = cp.cuda.runtime.getDeviceCount()
        ap_parray = None
        ap_list = None

        for k in range(num_tests):
            ap = a1.copy()

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

            if k == 0:
                ap_list = list()
                for i in range(n//block_size):
                    ap_list.append(list())
                    for j in range(n//block_size):
                        with cp.cuda.Device(i%n_gpus):
                            ap_list[i].append(cp.asarray(a1[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size], order='F'))
                            cp.cuda.Device().synchronize()
                ap_parray = asarray_batch(ap_list)
            else:
                rs = TaskSpace("Reset")
                for i in range(n//block_size):
                    for j in range(n//block_size):
                        @spawn(taskid=rs[i,j], placement=gpu(i%n_gpus), inout=[ap_parray[i][j]])
                        def reset():
                            ap_parray[i][j].update(cp.asarray(a1[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size], order='F'))
                            cp.cuda.stream.get_current_stream().synchronize()

                await rs

            print("Starting Cholesky")
            print("------------")
            start = time.perf_counter()

            # Call Parla Cholesky result and wait for completion
            await cholesky_blocked_inplace(ap_parray, block_size)
            print(ap_parray)

            #print(ap)
            end = time.perf_counter()

            ts = TaskSpace("CopyBack")
            @spawn(taskid=ts[0], placement=cpu, input=flatten(ap_parray))
            def copy_back():
                for i in range(n//block_size):
                    for j in range(n//block_size):
                        ap[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size] = ap_parray[i][j].array
            await ts

            if time_zeros:
                zerofy_start = time.perf_counter()
                computed_L_cupy = cp.tril(cp.array(ap))
                zerofy_end = time.perf_counter()
            else:
                zerofy_start = 0
                zerofy_end = 0

            print(f"Time:", (end - start) + (zerofy_end - zerofy_start))
            summarize_memory()
            clean_memory()
            print("--------")
            # Check result
            print("Is NAN: ", np.isnan(np.sum(ap)))
            if (np.isnan(np.sum(ap))) == True:
                print(ap)

            if check_error:
                if time_zeros:
                    computed_L = cp.asnumpy(computed_L_cupy)
                else:
                    computed_L = np.tril(ap)
                print(computed_L)
                error = np.max(np.absolute(a - computed_L @ computed_L.T))
                print("Error", error)

if __name__ == '__main__':
    np.random.seed(10)
    random.seed(10)
    with Parla():
        main()
