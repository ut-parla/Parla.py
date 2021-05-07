from concurrent.futures import ThreadPoolExecutor, wait
import traceback

import os
import sys
import time

OMP_NUM_THREADS = sys.argv[2]
nworkers = int(sys.argv[1])
n = 40000
nruns = 11

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

from test_data import discrete_laplacian

base_array = discrete_laplacian(n)

#print(sla.eigsh(base_array, 25, which = 'LM')[0])

# Store underlying buffers as memoryviews for handoff
# to different VECs.
data = memoryview(base_array.data)
indices = memoryview(base_array.indices)
indptr = memoryview(base_array.indptr)
np.random.seed(0)
v0 = memoryview(np.random.rand(n))

for i in range(nruns):
    start = time.perf_counter()
    for j in range(nworkers):
        a = sparse.csr_matrix((data, indices, indptr), shape = (n, n))
        eig = sla.eigsh(a, 25, which = 'LM', v0 = np.asarray(v0))
    stop = time.perf_counter()
    print(stop - start, flush = True)

