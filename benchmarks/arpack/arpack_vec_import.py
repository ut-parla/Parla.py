import os
import sys
import time

start = time.perf_counter()
from parla.multiload import multiload, multiload_contexts

nworkers = int(sys.argv[1])

for i in range(nworkers):
    with multiload_contexts[i] as VEC:
        #print("start setup for VEC {}".format(i), file = sys.stderr)
        import numpy as np
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sla
stop = time.perf_counter()
print(stop - start)
