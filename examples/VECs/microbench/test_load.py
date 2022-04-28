import time

t = time.perf_counter()
from parla.multiload import multiload_contexts
begin_multiload = time.perf_counter() - t

print(begin_multiload)


