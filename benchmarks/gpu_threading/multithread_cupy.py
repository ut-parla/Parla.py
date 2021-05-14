from threading import Thread, Barrier
import time
import argparse
import numpy as np
#from cuda_wrapper.core import *
import cupy as cp
from test import *
parser = argparse.ArgumentParser(description='Kokkos reduction')

parser.add_argument('-m', metavar='m', type=int,
                    help='Number of GPUs')

parser.add_argument('-trials', metavar='trials', type=int, help="Number of warmup runs")
args = parser.parse_args()


if __name__ == '__main__':

    m = args.m
    n_local = 1000000000
    N = m * n_local

    barrier = Barrier(m)
    t = time.time()
    #array = np.arange(1, N+1, dtype='float64')
    result = np.zeros(m, dtype='float64')

    list_cupy = []
    list_host = []
    for i in range(m):
        with cp.cuda.Device(i):
            list_cupy.append(cp.zeros(n_local, dtype=np.float64))
            list_host.append(np.arange(n_local, dtype=np.float64))


    t = time.time() - t
    print("Initilize array time: ", t, flush=True)


    def copy_task(d_arr, array, i):
        t = time.time()
        cpy_copy(d_arr, array, i)
        t = time.time() - t 
        print(i, "copy e2e time (per thread)", t)
        barrier.wait()
        t = time.time()
        cpy_copy(d_arr, array, i)
        t = time.time() - t 
        print(i, "copy e2e time (per thread) 2", t)
        barrier.wait()
        t = time.time()
        cpy_copy(d_arr, array, i)
        t = time.time() - t 
        print(i, "copy e2e time (per thread) 3", t)

    times = []
    for k in range(args.trials):
        t = time.time()
        threads = []
        for i in range(m):
            start = (i)*n_local
            end   = (i+1)*n_local
            y = Thread(target=copy_task, args=(list_cupy[i], list_host[i], i))
            threads.append(y)
        t = time.time() - t
        print("Initialize threads time: ", t, flush=True)
            
        t = time.time()
        for i in range(m):
            threads[i].start()

        for i in range(m):
            threads[i].join()
        t = time.time() - t
        print(f"Global E2E Time: {k}", t, flush=True)
        times.append(t)

    print("Median, mean, var", np.median(times), np.mean(times), np.var(times))

