import numpy as np
import cupy as cp
from parla.cuda import gpu
from parla.cpu import cpu
from parla.partitioning import partition1d_tensor
from parla.tasks import spawn, TaskSpace

a = np.random.rand(10000000)
b = np.random.rand(10000000)

divisions = 100

# Map the divisions onto actual hardware locations
def location(i):
    return cpu(0) if i < divisions // 2 else gpu(0)

def memory(i):
    return location(i).memory()

a_part = partition1d_tensor(divisions, a, memory = memory)
b_part = partition1d_tensor(divisions, b, memory = memory)

inner_result = np.empty(1)

@spawn(placement = cpu(0))
def inner_part():
    divisions = len(a_part)
    partial_sums = np.empty(divisions)
    partial_sum_tasks = TaskSpace("partial_sums")
    for i in range(divisions):
        @spawn(partial_sum_tasks[i], placement = location(i))
        def inner_local():
            partial_sums[i] = float(a_part[i] @ b_part[i])
    @spawn(None, [partial_sum_tasks[0:divisions]], placement = cpu(0))
    def accumulate_results():
        res = 0.
        for i in range(divisions):
            res += partial_sums[i]
        inner_result[0] = res

assert np.allclose(np.inner(a, b), inner_result[0])
