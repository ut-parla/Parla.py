import logging
# logging.basicConfig(level=logging.INFO)

#import parla.cpu
from parla.multiload import multiload, MultiloadContext
import timeit
import time

if __name__ == '__main__':
    #time.sleep(20)  #To give me enough time to attach strace
    m = 2
    def thing(i):
        n = 100000
        array = np.arange(1, n+1, dtype='float64')
        result = kokkos.reduction(array)
        return result

    for i in range(m):
        with MultiloadContext(i):
            import numpy as np
            import kokkos.gpu.core as kokkos
            kokkos.start(i)

    for i in range(m):
        with MultiloadContext(i):
            ctx = thing(i)
        print(i, ctx)

    for i in range(m):
        with MultiloadContext(i):
            kokkos.end(i)

