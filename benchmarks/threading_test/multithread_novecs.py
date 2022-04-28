from threading import Thread, Barrier
#from parla.multiload import multiload_contexts
import time
import argparse
import test.core as kokkos 
import numpy as np

parser = argparse.ArgumentParser(description='Kokkos reduction')

parser.add_argument('-m', metavar='m', type=int,
                    help='Number of GPUs')

parser.add_argument('-trials', metavar='trials', type=int, help="Number of warmup runs")
args = parser.parse_args()

if __name__ == '__main__':

    m = args.m
    n_local = 1000000000
    N = m * n_local

    b = Barrier(args.m)

    t = time.time()
    array = np.arange(1, N+1, dtype='float64')
    result = np.zeros(m, dtype='float64')
    t = time.time() - t
    print("Initilize array time: ", t, flush=True)


    def reduction(array, i):
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
        print(f"Time in threads: {k}", t, flush=True)
        times.append(t)

    print("Median, mean, var", np.median(times), np.mean(times), np.var(times))


