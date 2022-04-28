from threading import Thread
import time
import argparse
import numpy as np
from test.core import *
import cupy as cp
from mpi4py import MPI

parser = argparse.ArgumentParser(description='Kokkos reduction')

parser.add_argument('-m', metavar='m', type=int,
                    help='Number of GPUs')

parser.add_argument('-trials', metavar='trials', type=int, help="Number of warmup runs")
args = parser.parse_args()

if __name__ == '__main__':


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    m = args.m
    n_local = 1000000000
    N = m * n_local

    t = time.time()
    #array = np.arange(1, N+1, dtype='float64')
    result = np.zeros(m, dtype='float64')

    list_cupy = []
    list_host = []
    for i in range(m):
        with cp.cuda.Device(rank):
            list_cupy.append(cp.zeros(n_local, dtype=np.float64))
            list_host.append(np.arange(n_local, dtype=np.float64))


    t = time.time() - t
    print("Initilize array time: ", t, flush=True)


    def copy_task(d_arr, array, i):
        t = time.time()
        cpy_copy(d_arr, array, i)
        t = time.time() - t 
        print(i, "copy e2e time (per thread)", t)

    times = []
    for k in range(args.trials):
        t = time.time()
        i = rank
        copy_task(list_cupy[0], list_host[0], rank)
        t = time.time() - t
        print(f"Global E2E Time: {k}", t, flush=True)
        times.append(t)

    print("Median, mean, var", np.median(times), np.mean(times), np.var(times))
