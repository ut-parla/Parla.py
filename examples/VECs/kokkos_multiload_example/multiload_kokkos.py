import logging

#import parla.cpu
from parla.multiload import multiload_contexts
import time

if __name__ == '__main__':
    #time.sleep(20)  #To give me enough time to attach strace
    m = 2
    def thing(i):
        n = 100000
        array = np.arange(1, n+1, dtype='float64')
        result = kokkos.reduction(array, i)
        return result

    for i in range(m):
        multiload_contexts[i].load_stub_library("cuda")
        multiload_contexts[i].load_stub_library("cudart")
        with multiload_contexts[i]:
            a = 1
            import numpy as np
            import kokkos.gpu.core as kokkos
            kokkos.start(i)

    for i in range(m):
        with multiload_contexts[i]:
            ctx = thing(i)
        print(i, ctx)

    for i in range(m):
        with multiload_contexts[i]:
            b = 1
            kokkos.end()

