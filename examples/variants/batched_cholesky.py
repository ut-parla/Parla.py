import random
import os
import mkl
import argparse

threads = 6
mkl.set_num_threads(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=300, help='number of matrices')
parser.add_argument('-use_cpu', type=int, default=1, help='Use CPU?')
parser.add_argument('-ngpus', type=int, default=0, help='number of gpus')
parser.add_argument('-m_cpu', type=int, default=2000, help='number of rows for CPU task')
parser.add_argument('-m_gpu', type=int, default=2000, help='number of rows for GPU task ')
args = parser.parse_args()

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    print("CUDA_VISIBLE_DEVICES is not set. Assuming 0-3")
    cuda_visible_devices = list(range(4))
else:
    cuda_visible_devices = cuda_visible_devices.strip().split(',')
    cuda_visible_devices = list(map(int, cuda_visible_devices))

gpus = cuda_visible_devices[:max(1, args.ngpus)]

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

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
from parla.array import clone_here


if args.ngpus > 0:
    n_gpus = cp.cuda.runtime.getDeviceCount()
else:
    n_gpus = 0

np.random.seed(10)
cp.random.seed(10)
random.seed(10)

@specialized
def potrf(a):
    a = np.linalg.cholesky(a)
    #a = a @ a.T
    return a

@potrf.variant(gpu)
def potrf_gpu(a):
    stream = cp.cuda.get_current_stream()
    a = clone_here(a)
    a = cp.linalg.cholesky(a)
    #a = a @ a.T
    a = a.get()
    stream.synchronize()
    return a

def main():

    @spawn(placement=cpu)
    async def main_task():
        matrix_list = []

        #warmup
        for i in range(n_gpus):
            with cp.cuda.Device(i):
                a = cp.random.randn(args.m_gpu, args.m_gpu)
                A = a @ a.T + 1
                B = cp.linalg.cholesky(A)
                cp.cuda.Device(i).synchronize()

        for i in range(args.n):
            if i <= args.n//4:
                a = np.random.randn(args.m_gpu, args.m_gpu)
                A = a @ a.T + 1
            else:
                a = np.random.randn(args.m_cpu, args.m_cpu)
                A = a @ a.T + 1

            matrix_list.append(A)

        cp.cuda.Device().synchronize()
        random.shuffle(matrix_list)
        print("Finished setting up matrices.")
        start_external_t = time.perf_counter()
        TS = TaskSpace("TaskSpace")
        i = 0
        for A in matrix_list:
            i+=1

            loc = []
            if args.use_cpu:
                loc.extend([cpu])
            loc.extend([gpu(i) for i in range(n_gpus)])

            load=0.5

            @spawn(TS[i], placement=loc, vcus=load)
            def matrix_task():
                start_t = time.perf_counter()
                B = potrf(A)
                end_t = time.perf_counter()

        await TS
        end_external_t = time.perf_counter()
        print("Time: ", end_external_t - start_external_t, flush=True)

if __name__ == '__main__':
    with Parla():
        main()
