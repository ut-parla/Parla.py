import logging
# logging.basicConfig(level=logging.INFO)

#import parla.cpu
from parla.multiload import multiload, multiload_contexts
import timeit
import time

if __name__ == '__main__':
    #time.sleep(30)  #To give me enough time to attach strace
    m = 5
    t = 1
    def thing(i):
        n = 10000
        array = 2*np.arange(1, n+1, dtype='float32')
        out = np.arange(1, n+1, dtype='float32')
        result = kokkos.add_vectors(array, array, out)
        #result = kokkos.reduction(a2, i)
        return result

    for i in range(m):
        multiload_contexts[i].load_stub_library("cuda")
        multiload_contexts[i].load_stub_library("cudart")
        with multiload_contexts[i]:
            import numpy as np
            import kokkos.gpu.core as kokkos
            #kokkos.start(i)

    for i in range(m):
        with multiload_contexts[i]:
            ctx = thing(i)
        print(i, ctx)

    #for i in range(t, m):
    #    with multiload_contexts[i]:
    #        kokkos.end()

