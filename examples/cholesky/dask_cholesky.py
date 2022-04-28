import time

import numpy
import cupy
import dask
import dask.array
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

# if __name__ == __main__ block needed to avoid
# some wierdness with multiprocessing. See
# https://github.com/dask/distributed/issues/2520#issuecomment-470817810
if __name__ == '__main__':
    cluster = LocalCUDACluster()
    client = Client(cluster)
    n = 40000
    block_size = 2000
    numpy.random.seed(10)
    a = numpy.random.rand(n, n)
    a = a @ a.T
    a = dask.array.from_array(a, chunks = (block_size, block_size))
    a = a.map_blocks(cupy.asarray)
    a.compute()
    start = time.perf_counter()
    cho = dask.array.linalg.cholesky(a)
    cho.compute()
    stop = time.perf_counter()
    print(stop - start)
