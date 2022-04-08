import time

start = time.perf_counter()
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

from test_data import discrete_laplacian
stop = time.perf_counter()
print(stop - start)

