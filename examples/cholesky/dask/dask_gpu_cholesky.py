import os
import mkl
import argparse
import operator
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('-workers', type=int, default=1)
parser.add_argument('-t', type=int, default=1)
parser.add_argument('-matrix', default=None)
parser.add_argument('-process', type=int, default=0)
args = parser.parse_args()

is_process = True if args.process else False

mkl.set_num_threads(args.t)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.t)
os.environ["OMP_NUM_THREADS"] = str(args.t)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.t)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.t)

import pprint
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.array.core import Array
from dask.array.utils import array_safe, meta_from_array, solve_triangular_safe
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.diagnostics import ProgressBar

import dask.array
import dask
import numpy as np
import cupy as cp
from cupy.cuda import cublas
from cupy.cuda import device
from scipy.linalg.blas import dtrsm

import time

def potrf(a):
    a = cp.linalg.cholesky(a)
    return a


def cupy_trsm_wrapper(a, b):
    cublas_handle = device.get_cublas_handle()
    trsm = cublas.dtrsm
    uplo = cublas.CUBLAS_FILL_MODE_LOWER

    a = cp.asarray(a, dtype=np.float64, order='F')
    b = cp.asarray(b, dtype=np.float64, order='F')

    trans = cublas.CUBLAS_OP_T
    side = cublas.CUBLAS_SIDE_RIGHT

    diag = cublas.CUBLAS_DIAG_NON_UNIT
    m, n = (b.side, 1) if b.ndim == 1 else b.shape
    trsm(cublas_handle, side, uplo, trans, diag, m, n, 1.0, a.data.ptr, m, b.data.ptr, m)
    return b


def solve(a, b):
    b = cupy_trsm_wrapper(a, b)
    return b

def handle_trsm(a, b):
    b = solve(a, b)
    return b


def gemm(a, b):
    return a@b.T


def syrk(a):
    return a@a.T


def zerofy(a):
    return partial(cp.zeros_like, shape=(a.chunks[0][i], a.chunks[1][j]))


def blocked_cholesky(a):
    num_blocks = len(a.chunks[0])
    token = tokenize(a)
    name = "cholesky-" + token
    name_lt_dot = "cholesky-lt-dot-" + token

    dsk = {}
    print("num_blocks:", num_blocks)
    for i in range(num_blocks):
        if i > 0:
            prevs = []
            for k in range(i):
                prev = name_lt_dot, i, k, i, k
                dsk[prev] = (operator.sub, (a.name, i, i), (syrk, (name, i, k)))
                prevs.append(prev)
            dsk[name, i, i] = (potrf, (sum, prevs))
        else:
            dsk[name, i, i] = (potrf, (a.name, i, i))
        for j in range(i+1, num_blocks):
            if i > 0:
                prevs = []
                for k in range(i):
                    prev = name_lt_dot, j, k, i, k
                    dsk[prev] = (operator.sub, (a.name, j, i), (gemm, (name, j, k), (name, i, k)))
                    prevs.append(prev)
                dsk[name, j, i] = (handle_trsm, (name, i, i), (sum, prevs))
            else:
                dsk[name, j, i] = (handle_trsm, (name, i, i), (a.name, j, i))

    # Zerofy the upper matrix.
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i < j:
                dsk[name, i, j] = (
                     partial(np.zeros_like, shape=(a.chunks[0][i], a.chunks[1][j])),
                     meta_from_array(a),
                )

    graph_lower = HighLevelGraph.from_collections(name, dsk, dependencies=[a])
    a_meta = meta_from_array(a)
    cho = np.linalg.cholesky(array_safe(
        [[1, 2], [2, 5]], dtype=a.dtype, like=a_meta))
    meta = meta_from_array(a, dtype=cho.dtype)

    lower = Array(graph_lower, name, shape=a.shape, chunks=a.chunks, meta=meta)
    #lower.visualize(filenmae='check.svg')
    return lower

if __name__ == '__main__':
    cluster=LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1,2,3",
                             enable_nvlink=True,
                             n_workers=4,
                             protocol="ucx",
                             interface="ib0",
                             enable_tcp_over_ucx=True,
                             rmm_pool_size="3GB"
                            )
    client = Client(cluster)
    if args.matrix is None:
        n = 20000
        block_size = 2000
        np.random.seed(10)
        a = np.random.rand(n, n)
        a = a @ a.T
    else:
        block_size = 2000
        print("Loading matrix from file: ", args.matrix)
        a = np.load(args.matrix)
        print("Loaded matrix from file. Shape=", a.shape)
        n = a.shape[0]

    da = dask.array.from_array(a, chunks='auto')#chunks=(block_size, block_size))
    da = da.map_blocks(cp.asarray)
    chol = blocked_cholesky(da)
    start = time.perf_counter()
    out_cp = chol.compute()
    stop = time.perf_counter()
    print("Time: ", stop - start)

    ErrorCheck = True
    if ErrorCheck == True:
        out = np.asarray(out_cp.get())
        error = np.max(np.absolute(a - out @out.T))
        print("Error:", error)

