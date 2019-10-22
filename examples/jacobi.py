import logging

import numpy as np
import numba.cuda

from parla.array import copy
from parla.cuda import gpu
from parla.cpu import cpu
from parla.function_decorators import specialized
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import spawn, TaskSpace, CompletedTaskSpace
import parla.array

import cupy

import os
import time

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# parla.array.logger.setLevel(logging.DEBUG)
# parla.cuda.logger.setLevel(logging.DEBUG)
# parla._cpuutils.logger.setLevel(logging.DEBUG)

# CPU code to perform a single step in the Jacobi iteration.
# Specialized later by jacobi_gpu
@specialized
@numba.njit(parallel=True)
def jacobi(a0, a1):
    a1[1:-1,1:-1] = .25 * (a0[2:,1:-1] + a0[:-2,1:-1] + a0[1:-1,2:] + a0[1:-1,:-2])

# Actual cuda kernel to do a single step
@numba.cuda.jit
def gpu_jacobi_kernel(a0, a1):
    i, j = numba.cuda.grid(2)
    if 0 < i < a1.shape[0]-1 and 0 < j < a1.shape[1]-1:
        a1[i,j] = .25 * (a0[i-1,j] + a0[i+1,j] + a0[i,j-1] + a0[i,j+1])

# GPU kernel call to perform a single step in the Jacobi iteration.
@jacobi.variant(gpu)
def jacobi_gpu(a0, a1):
    threads_per_block_x = 32
    threads_per_block_y = 1024//threads_per_block_x
    blocks_per_grid_x = (a0.shape[0] + (threads_per_block_x - 1)) // threads_per_block_x
    blocks_per_grid_y = (a0.shape[1] + (threads_per_block_y - 1)) // threads_per_block_y
    # Relying on numba/cupy interop here.
    gpu_jacobi_kernel[(blocks_per_grid_x,blocks_per_grid_y), (threads_per_block_x,threads_per_block_y)](a0, a1)
"""
# Actual cuda kernel to do a single step
@numba.cuda.jit
def gpu_jacobi_kernel(a0, a1):
    start = numba.cuda.grid(1)
    stride = numba.cuda.gridsize(1)
    for i in range(start + 1, a0.shape[0] - 1, stride):
        for j in range(1, a1.shape[1] - 1):
            a1[i,j] = .25 * (a0[i-1,j] + a0[i+1,j] + a0[i,j-1] + a0[i,j+1])


# GPU kernel call to perform a single step in the Jacobi iteration.
@jacobi.variant(gpu)
def jacobi_gpu(a0, a1):
    threads_per_block = 128
    blocks_per_grid = (a0.shape[0] + (threads_per_block - 1)) // threads_per_block
    # Relying on numba/cupy interop here.
    gpu_jacobi_kernel[blocks_per_grid, threads_per_block](a0, a1)
"""

def main():
    devs = list(gpu.devices) + list(cpu.devices)
    if "N_DEVICES" in os.environ:
        devs = devs[:int(os.environ.get("N_DEVICES"))]
    divisions = len(devs) * 2

    # Set up an "n" x "n" grid of values and run
    # "steps" number of iterations of the 4 point stencil on it.
    # n = int(25000*divisions**0.5)
    #n = 36
    #steps = 6
    #overlap = 3
    n = 25000
    steps = 200
    overlap = 10

    # Set up two arrays containing the input data.
    # This demo uses the standard technique of computing
    # from one array into another then swapping the
    # input and output arrays for the next iteration.
    # These are the two arrays that will be swapped back
    # and forth as input and output.
    a0 = np.random.rand(n, 4)
    a1 = a0.copy()

    #actual = a1.copy()
    #for i in range(steps):
    #    actual[1:-1,1:-1] = .25 * (actual[2:,1:-1] + actual[:-2,1:-1] + actual[1:-1,2:] + actual[1:-1,:-2])

    # An object that distributes arrays across all the given devices.
    mapper = LDeviceSequenceBlocked(divisions, devices=devs)
    # print(mapper.devices)

    # Partition a0 and a1.
    # Here we just partition the rows across the different devices.
    # Other partitioning schemes are possible.
    a0_row_groups = mapper.partition_tensor(a0, overlap=overlap)
    a1_row_groups = mapper.partition_tensor(a1, overlap=overlap)

    # Trigger JIT
    start = time.perf_counter()
    @spawn(placement=cpu(0))
    async def warmups():
        warmup = TaskSpace()
        for i in range(divisions):
            @spawn(warmup[i], placement=mapper.device(i))
            async def w():
                jacobi(a0_row_groups[i], a1_row_groups[i])
                cupy.cuda.get_current_stream().synchronize()
                cupy.cuda.Stream.null.synchronize()
        await warmup
    end = time.perf_counter()
    # print("warmup", end - start)

    start = time.perf_counter()
    # Main parla task.
    @spawn(placement=cpu(0))
    async def run_jacobi():
        assert steps > 0
        assert not steps % overlap
        # Specify which set of blocks is used as input or output
        # (they will be swapped for each iteration).
        in_blocks = a0_row_groups
        out_blocks = a1_row_groups
        # Create a set of labels for the tasks that perform the first
        # Jacobi iteration step.
        previous_communication_tasks = CompletedTaskSpace()
        # Now create the tasks for subsequent iteration steps.
        for i in range(0, steps, overlap):
            # Create a new set of labels for the tasks that do this iteration step.
            current_block_tasks = TaskSpace("block_tasks[{}]".format(i))
            current_communication_tasks = TaskSpace("communcation[{}]".format(i))
            # Create the tasks to do the i'th iteration.
            # As before, each task needs the following info:
            #  a block index "j"
            #  a "device" where it should execute (supplied by mapper used for partitioning)
            #  the "in_block" of data used as input
            #  the "out_block" to write the output to
            for j in range(divisions):
                device = mapper.device(j)
                in_block = in_blocks[j]
                out_block = out_blocks[j]
                # Make each task operating on each block depend on the tasks for
                # that block and its immediate neighbors from the previous iteration.
                @spawn(current_block_tasks[j],
                       dependencies=[previous_communication_tasks[max(0, j-1):min(divisions, j+2)]],
                       placement=device)
                def device_local_jacobi_task():
                    # c_end = time.perf_counter()
                    # print(device, j, i, "copy", c_end - c_start, in_block[0].shape)
                    # Run the computation, dispatching to device specific code.
                    # w_start = time.perf_counter()
                    local_in = in_block
                    local_out = out_block
                    for k in range(overlap):
                        lower = k
                        upper = local_in.shape[0] - k
                        if j == 0:
                            lower = 0
                        if j == divisions - 1:
                            upper = local_in.shape[0]
                        jacobi(local_in[lower:upper], local_out[lower:upper])
                        local_in, local_out = local_out, local_in
                    # cupy.cuda.get_current_stream().synchronize()
                    # w_end = time.perf_counter()
                    # print(device, j, i, "work", w_end - w_start, in_block.shape)
            if overlap % 2:
                in_blocks, out_blocks = out_blocks, in_blocks
            if i + overlap >= steps:
                previous_communication_tasks = current_block_tasks
                break
            for j in range(divisions):
                device = mapper.device(j)
                @spawn(current_communication_tasks[j],
                       dependencies=[current_block_tasks[max(0, j-1):min(divisions, j+2)]],
                       placement=device)
                def device_local_data_movement():
                    if j > 0:
                        copy(in_blocks[j][:overlap], in_blocks[j - 1][-2*overlap:-overlap])
                    if j < divisions - 1:
                        copy(in_blocks[j][-overlap:], in_blocks[j + 1][overlap:2*overlap])
            # For the next iteration, use the newly created tasks as
            # the tasks from the previous step.
            previous_communication_tasks = current_communication_tasks
        await previous_communication_tasks
        end = time.perf_counter()
        print(end - start)
        # This depends on all the tasks from the last iteration step.
        for j in range(divisions):
            start_index = overlap if j > 0 else 0
            end_index = -overlap if j < divisions - 1 else None  # None indicates the last element of the dimension
            copy(a1[mapper.slice(j, len(a1))], in_blocks[j][start_index:end_index])
        #assert(np.absolute(actual - a1).max() < 1E-7)


if __name__ == '__main__':
    # import yappi
    #
    # yappi.start()

    main()

    # yappi.get_func_stats().print_all()
    # yappi.get_thread_stats().print_all()
