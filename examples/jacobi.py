import numpy as np
import numba as nb
import cupy as cp
from numba import cuda
import parla as pl
from parla.function_decorators import specialized
from parla.cuda import gpu
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace

@specialized
@nb.njit(parallel = True)
def jacobi(a0, a1):
    a1[1:-1,1:-1] = .25 * (a0[2:,1:-1] + a0[:-2,1:-1] + a0[1:-1,2:] + a0[1:-1,:-2])

@cuda.jit
def gpu_jacobi_kernel(a0, a1):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start + 1, a0.shape[0] - 1, stride):
        for j in range(1, a1.shape[1] - 1):
            a1[i,j] = .25 * (a0[i-1,j] + a0[i+1,j] + a0[i,j-1] + a0[i,j+1])

@jacobi.variant(gpu)
def jacobi_gpu(a0, a1):
    threads_per_block = 64
    blocks_per_grid = (a0.shape[0] + (threads_per_block - 1)) // threads_per_block
    # Relying on numba/cupy interop here.
    gpu_jacobi_kernel[blocks_per_grid, threads_per_block](a0, a1)

n = 4000
steps = 6

a0 = np.random.rand(n, n)
a1 = a0.copy()

actual = a0.copy()
for i in range(steps):
    actual[1:-1,1:-1] = .25 * (actual[2:,1:-1] + actual[:-2,1:-1] + actual[1:-1,2:] + actual[1:-1,:-2])

divisions = 40
rows_per_division = (n + divisions - 1) // divisions

split = divisions // 2
def location(i):
    return cpu(0) if i < split else gpu(0)

locations = [location(i) for i in range(divisions)]
a0_row_groups = [locations[i].memory()(a0[max(0, i * rows_per_division - 1):min(n, (i+1) * rows_per_division + 1)])
                 for i in range(divisions)]
a1_row_groups = [locations[i].memory()(a1[max(0, i * rows_per_division - 1):min(n, (i+1) * rows_per_division + 1)])
                 for i in range(divisions)]

@spawn(placement=cpu(0))
def run_jacobi():
    assert steps > 0
    in_blocks = a0_row_groups
    out_blocks = a1_row_groups
    previous_block_tasks = TaskSpace("block_tasks[0]")
    for j, (location, in_block, out_block) in enumerate(zip(locations, in_blocks, out_blocks)):
        @spawn(previous_block_tasks[j], placement=location)
        def device_local_jacobi_task():
            jacobi(in_block, out_block)
    for i in range(1, steps):
        in_blocks, out_blocks = out_blocks, in_blocks
        current_block_tasks = TaskSpace("block_tasks[{}]".format(i))
        for j, (location, in_block, out_block) in enumerate(zip(locations, in_blocks, out_blocks)):
            @spawn(current_block_tasks[j],
                   [previous_block_tasks[max(0, j-1):min(divisions, j+2)]],
                   placement=location)
            def device_local_jacobi_task():
                # communication
                if j > 0:
                    in_block[0] = location.memory()(in_blocks[j-1][-2])
                if j < divisions - 1:
                    in_block[-1] = location.memory()(in_blocks[j+1][1])
                # computation
                jacobi(in_block, out_block)
        previous_block_tasks = current_block_tasks
    # Aggregate the results into original a1 buffer
    @spawn(None, [previous_block_tasks[0:divisions]], placement = cpu(0))
    def aggregate():
        for j in range(divisions):
            start_index = 1 if j > 0 else 0
            end_index = -1 if j < divisions - 1 else rows_per_division + 1
            a1[j * rows_per_division:(j+1) * rows_per_division] = cpu(0).memory()(out_blocks[j][start_index:end_index])

assert np.max(np.absolute(a1 - actual)) < 1E-14
