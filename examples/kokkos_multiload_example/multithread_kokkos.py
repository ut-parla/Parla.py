from threading import Thread
from parla.multiload import multiload_contexts
import time

if __name__ == '__main__':

    m = 4
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
            import kokkos.gpu.core as kokkos
            kokkos.start(i)
    t = time.time() - t
    print("Initialize time: ", t, flush=True)

    array = np.arange(1, N+1, dtype='float64')
    result = np.zeros(m, dtype='float64')

    def reduction(array, i):
        global n_local
        start = (i)*n_local
        end   = (i+1)*n_local
        with multiload_contexts[i]:
            result[i] = kokkos.reduction(array[start:end], i)

    threads = []
    for i in range(m):
        y = Thread(target=reduction, args=(array, i))
        threads.append(y)
        y.start()

    for i in range(m):
        threads[i].join()

    result = np.sum(result)
    print("Final Result: ", result, (N*(N+1))/2, flush=True)


    t = time.time()
    for i in range(m):
        with multiload_contexts[i]:
            kokkos.end()
    t = time.time() - t
    print("Finalize Time: ", t, flush=True)


