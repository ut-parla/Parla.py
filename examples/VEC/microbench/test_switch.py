import time

t = time.perf_counter()
from parla.multiload import multiload_contexts
begin_multiload = time.perf_counter() - t

m = 12
import_times = []
for i in range(m):
    with multiload_contexts[i]:
        t = time.perf_counter()
        import numpy as np
        t = time.perf_counter() - t
        import_times.append(t)

with multiload_contexts[0]:
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)

switch_times = []
for i in range(m):
    t = time.perf_counter()
    with multiload_contexts[i]:
        t = time.perf_counter() - t
        c = a @ b
    switch_times.append(t)

print(np.mean(switch_times))



