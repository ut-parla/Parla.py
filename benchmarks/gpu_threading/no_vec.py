from threading import Thread
import time
import argparse


parser = argparse.ArgumentParser(description='Kokkos reduction')

parser.add_argument('-m', metavar='m', type=int,
                    help='Number of GPUs')

parser.add_argument('-trials', metavar='trials', type=int, help="Number of warmup runs")
args = parser.parse_args()

if __name__ == '__main__':

    m = args.m
    n_local = 10000**2
    N = m * n_local

    
    t = time.time()
    import numpy as np
    import kokkos.gpu.core as kokkos
    kokkos.start(m)
    t = time.time() - t
    print("Initilize time: ", t, flush=True)

    t = time.time()
    array = np.arange(1, N+1, dtype='float64')
    result = np.zeros(m, dtype='float64')
    t = time.time() - t
    print("Initilize array time: ", t, flush=True)

    def reduction(array, i):

        p = kokkos.dev_copy(array, (int)(np.sqrt(len(array))), i)
        result = kokkos.reduction(p, (int)(np.sqrt(len(array))), i)

    times = []
    for k in range(args.trials):
        t = time.time()
        threads = []
        for i in range(1):
            start = (i)*n_local
            end   = (i+1)*n_local
            y = Thread(target=reduction, args=(array[start:end], m))
            threads.append(y)
        t = time.time() - t
        print("Initialize threads time: ", t, flush=True)
            
        t = time.time()
        for i in range(1):
            threads[i].start()

        for i in range(1):
            threads[i].join()

        t = time.time() - t
        print(f"Reduction Time: {k}", t, flush=True)
        times.append(t)

    times = np.array(times)
    times = times[1:]
    print("Median, mean, var", np.median(times), np.mean(times), np.var(times))

    t = time.time()
    s = 0.0
    for i in range(m):
        s += result[i]
    result = s
    #result = np.sum(result)
    t = time.time() - t
    print("Sum Time: ", t, flush=True)

    print("Final Result: ", result, (N*(N+1))/2, flush=True)
    
    t = time.time()
    for i in range(m):
        kokkos.end()
    t = time.time() - t
    print("Finalize Time: ", t, flush=True)


