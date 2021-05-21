from threading import Thread
from parla.multiload import multiload_contexts
import time
import argparse


parser = argparse.ArgumentParser(description='Kokkos reduction')

parser.add_argument('-m', metavar='m', type=int,
                    help='Number of GPUs')

parser.add_argument('-trials', metavar='trials', type=int, help="Number of warmup runs")
args = parser.parse_args()

if __name__ == '__main__':

    m = args.m
    n_local = 100000000
    N = m * n_local

    #Load and configure
    #Sequential to avoid numpy bug
    t = time.time()
    for i in range(m):
        multiload_contexts[i].load_stub_library("cuda")
        multiload_contexts[i].load_stub_library("cudart")
        with multiload_contexts[i]:
            import numpy as np
            import test.core as kokkos
    t = time.time() - t
    print("Initialize time: ", t, flush=True)

    t = time.time()
    array = np.arange(1, N+1, dtype='float64')
    result = np.zeros(m, dtype='float64')
    t = time.time() - t
    print("Initilize array time: ", t, flush=True)


    def reduction(array, i):
        print("ID: ", i, flush=True)
        global n_local
        start = (i)*n_local
        end   = (i+1)*n_local
        with multiload_contexts[i]:
            t = time.perf_counter()
            p = kokkos.dev_copy(array, (int)(np.sqrt(len(array))), i)
            t2 = time.perf_counter()
            b.wait()
            kokkos.clean(p, i)
        print("Copy: ", t2 -t, flush=True)

    times = []
    for k in range(args.trials):
        t = time.time()
        threads = []
        for i in range(m):
            start = (i)*n_local
            end   = (i+1)*n_local
            y = Thread(target=reduction, args=(array[start:end], i))
            threads.append(y)
        t = time.time() - t
        print("Initialize threads time: ", t, flush=True)
            
        t = time.time()
        for i in range(m):
            threads[i].start()

        for i in range(m):
            threads[i].join()

        t = time.time() - t
        print(f"Time in Threads: {k}", t, flush=True)
        times.append(t)

    times = np.array(times)
    times = times[1:]
    print("Median, mean, var", np.median(times), np.mean(times), np.var(times))


