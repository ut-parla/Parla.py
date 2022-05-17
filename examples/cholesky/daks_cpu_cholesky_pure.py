from dask.array import linalg as dsk 
from concurrent.futures import ThreadPoolExecutor

from dask.distributed import Client, LocalCluster
from dask.array.utils import array_safe, meta_from_array, solve_triangular_safe

import os
import dask.array as da
import argparse
import numpy as np
import mkl

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

parser = argparse.ArgumentParser()
parser.add_argument('-workers', type=int, default=1)
parser.add_argument('-perthread', type=int, default=1)
parser.add_argument('-t', type=int, default=1)
parser.add_argument('-process', type=int, default=0)
args = parser.parse_args()

is_process = True if args.process else False

mkl.set_num_threads(args.t)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.t)
os.environ["OMP_NUM_THREADS"] = str(args.t)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.t)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.t)


if __name__ == '__main__':
    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=args.perthread, processes=False)
    client = Client(cluster)
    n = 20000
    block_size = 2000
    np.random.seed(10)
    a = np.random.rand(n, n)
    a = a @ a.T
    da = da.from_array(a, chunks=(block_size, block_size))
    chol = dsk.cholesky(da, lower = True) 
    start = time.perf_counter()
    print(chol.compute())
    stop = time.perf_counter()
    print(stop - start)
