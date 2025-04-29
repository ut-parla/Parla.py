import time

t = time.perf_counter()
from parla.multiload import multiload_contexts
begin_multiload = time.perf_counter() - t

m = 4
import_times = []
for i in range(m):
    with multiload_contexts[i]:
        t = time.perf_counter()
        import numpy
        t = time.perf_counter() - t
        print(t)


