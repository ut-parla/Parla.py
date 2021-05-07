from concurrent.futures import ThreadPoolExecutor, wait
import traceback

import os
import sys
import time

from parla.multiload import multiload, multiload_contexts

OMP_NUM_THREADS = sys.argv[2]
nworkers = int(sys.argv[1])
n = 10000
nruns = 11

os.environ['OMP_NUM_THREADS'] = OMP_NUM_THREADS

#print("starting setup", file = sys.stderr)
#print("starting setup")
for i in range(nworkers):
    with multiload_contexts[i] as VEC:
        #print("start setup for VEC {}".format(i), file = sys.stderr)
        VEC.setenv("OMP_NUM_THREADS", OMP_NUM_THREADS)
        import numpy as np
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sla
        #print("end setup for VEC {}".format(i), file = sys.stderr)

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

pool = ThreadPoolExecutor(max_workers = nworkers)
#print("starting")
for i in range(nruns):
    def call_arpack(i):
        try:
            with multiload_contexts[i]:
                a = sparse.csr_matrix((data, indices, indptr), shape = (n, n))
                eig = sla.eigsh(a, 25, which = 'LM', v0 = np.asarray(v0))
        except:
            traceback.print_exc()
            raise

    start = time.perf_counter()
    futures = [pool.submit(call_arpack, i) for i in range(nworkers)]
    wait(futures)
    stop = time.perf_counter()
    print(stop - start, flush = True)

