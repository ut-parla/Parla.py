import random
import os
import mkl

threads = 6
mkl.set_num_threads(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)

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

import argparse

from parla.array import clone_here

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=20)
parser.add_argument('-mc', type=int, default=1000)
parser.add_argument('-mg', type=int, default=2000)
args = parser.parse_args()

n_gpus = cp.cuda.runtime.getDeviceCount()

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
                a = cp.random.randn(args.mg, args.mg)
                A = a @ a.T + 1
                B = cp.linalg.cholesky(A)
                cp.cuda.Device(i).synchronize()

        for i in range(args.n):
            if i <= args.n//4:
                a = np.random.randn(args.mg, args.mg)
                A = a @ a.T + 1
            else:
                a = np.random.randn(args.mc, args.mc)
                A = a @ a.T + 1

            matrix_list.append(A)

        cp.cuda.Device().synchronize()
        random.shuffle(matrix_list)
        start_external_t = time.perf_counter()
        TS = TaskSpace("TaskSpace")
        i = 0
        for A in matrix_list:
            i+=1

            if A.shape[0] < 10000:
                loc=[cpu]
                load=1
                #loc=[gpu(0)]
                #load=1
            else:
                loc=[cpu, gpu(0)]
                load=1/3

            @spawn(TS[i], placement=loc, vcus=load)
            def matrix_task():
                start_t = time.perf_counter()
                B = potrf(A)
                end_t = time.perf_counter()
                print(A.shape, type(B), end_t - start_t, flush=True)

        await TS
        end_external_t = time.perf_counter()
        print("Done.", end_external_t - start_external_t, flush=True)

if __name__ == '__main__':
    with Parla():
        main()

