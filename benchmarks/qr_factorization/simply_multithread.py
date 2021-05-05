import mkl
mkl.set_num_threads(6) # NTHREADS

import time 
import numpy as np
import scipy.linalg
import sys
import threading 

NGROUPS = 4

NROWS = 80000
NCOLS = 20000
np.random.seed(0)
a = np.random.rand(NROWS, NCOLS).astype(np.float32)

a_part = []
block_size = NROWS // NGROUPS + 1
for i in range(NGROUPS):
    a_part.append(np.array(a[i * block_size : (i + 1) * block_size]))

def worker(block):
    A = np.linalg.qr(block)
    return 

thread_list = []
for i in range(NGROUPS):
    t = threading.Thread(target=worker, args=(a_part[i],))
    thread_list.append(t)

print("Starting threads")
for t in thread_list:
    t.start()

for t in thread_list:
    t.join()

print("Done")
