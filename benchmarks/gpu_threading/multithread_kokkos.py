from threading import Thread, Barrier
from parla.multiload import multiload_contexts
import time
import argparse

parser = argparse.ArgumentParser(description='Kokkos reduction')

parser.add_argument('-m', metavar='m', type=int,
                    help='Number of GPUs')

parser.add_argument('-trials', metavar='trials', type=int, help="Number of warmup runs")
args = parser.parse_args()

b = Barrier(args.m)

if __name__ == '__main__':

    m = args.m
    n_local = 10000**2
    N = m * n_local
    
    #Load and configure
    #Sequential to avoid numpy bug
    t = time.time()
    for i in range(m):
        #multiload_contexts[i].load_stub_library("cuda")
        #multiload_contexts[i].load_stub_library("cudart")
        multiload_contexts[i]._set_pid_mutilation_number(i)
        print(multiload_contexts[i].__index__())
        with multiload_contexts[i]:
            import test.core as kokkos
            import numpy as np
            kokkos.start(i)
    t = time.time() - t
    print("Initialize time: ", t, flush=True)

    t = time.time()
    array = np.arange(1, N+1, dtype='float64')
    result = np.zeros(m, dtype='float64')
    t = time.time() - t
    print("Initilize array time: ", t, flush=True)


    def init(i):
        with multiload_contexts[i]:
            multiload_contexts[i]._set_pid_mutilation_number(i)
            #kokkos.start(i)

    def reduction(array, i):
        with multiload_contexts[i]:
            #t = time.perf_counter()
            p = kokkos.dev_copy(array, (int)(np.sqrt(len(array))), i)
            #t2 = time.perf_counter()
            #print("Kokkos Copy: ", t2 -t, flush=True)

            b.wait()

            #t = time.perf_counter()
            kokkos.reduction(p, (int)(np.sqrt(len(array))), i)
            #t2 = time.perf_counter()
            #print("Kokkos Kernel: ", t2 - t, flush=True)

    threads = []
    t = time.perf_counter()
    for i in range(m):
        start = (i)*n_local
        end   = (i+1)*n_local
        y = Thread(target=init, args=(i,))
        threads.append(y)
    t = time.perf_counter() - t
    print("Initialize threads time: ", t, flush=True)
        
    t = time.perf_counter()
    for i in range(m):
        threads[i].start()

    for i in range(m):
        threads[i].join()
    t = time.perf_counter() - t
    print("Initialize Kokkos time: ", t, flush=True)

    times = []
    for k in range(args.trials):

        threads = []
        t = time.perf_counter()
        for i in range(m):
            start = (i)*n_local
            end   = (i+1)*n_local
            y = Thread(target=reduction, args=(array[start:end], i))
            threads.append(y)
        t = time.perf_counter() - t
        print("Initialize threads time: ", t, flush=True)
            
        t = time.perf_counter()
        for i in range(m):
            threads[i].start()

        for i in range(m):
            threads[i].join()

        t = time.perf_counter() - t
        print(f"Reduction Time: {k}", t, flush=True)
        times.append(t)

    times = np.array(times)
    times = times[1:]
    print("Median, mean, var", np.median(times), np.mean(times), np.var(times))
    
    print("Final Result: ", result, flush=True)
    
    t = time.time()
    for i in range(m):
        with multiload_contexts[i]:
            kokkos.end()
    t = time.time() - t
    print("Finalize Time: ", t, flush=True)


