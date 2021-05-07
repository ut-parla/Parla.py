from concurrent.futures import ThreadPoolExecutor, wait
import traceback

import os
import sys
import time

OMP_NUM_THREADS = sys.argv[1]
n = 10000
nruns = 11

from mpi4py import MPI
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

from test_data import discrete_laplacian

# TODO: time shipping the input out to different processes?
base_array = discrete_laplacian(n)

#print(sla.eigsh(base_array, 25, which = 'LM')[0])

# Store underlying buffers as memoryviews for handoff
# to different VECs.
data = memoryview(base_array.data)
indices = memoryview(base_array.indices)
indptr = memoryview(base_array.indptr)
np.random.seed(0)
v0 = memoryview(np.random.rand(n))

rank = MPI.COMM_WORLD.Get_rank()
for i in range(nruns):
    MPI.COMM_WORLD.Barrier()
    start_time = MPI.Wtime()
    a = sparse.csr_matrix((data, indices, indptr), shape = (n, n))
    eig = sla.eigsh(a, 25, which = 'LM', v0 = np.asarray(v0))
    MPI.COMM_WORLD.Barrier()
    end_time = MPI.Wtime()
    if rank == 0:
        print(end_time - start_time)

