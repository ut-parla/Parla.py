ITERS = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 100, 100, 100, 100]
SIZES = [2**i for i in range(13)]
NTHREADS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]

import mkl
import numpy as np
from time import perf_counter as time

for i, s in enumerate(SIZES):
    mats = []
    for j in range(ITERS[i]):
        mats.append(np.random.rand(s, s))

    for t in NTHREADS:
        out = [None] * ITERS[i]
        mkl.set_num_threads(t)
        start = time()
        for j in range(ITERS[i]):
            out[j] = np.matmul(mats[j], mats[j])
        total = time() - start

        #print(s, ',', t, ',', total / ITERS[i], sep="")
        print(total / ITERS[i], ',', sep='', end='')
    print()
