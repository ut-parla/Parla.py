import numpy as np
import numba.cuda

from parla.array import copy
from parla.cuda import gpu
from parla.cpu import cpu
from parla.function_decorators import specialized
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import spawn, TaskSpace, CompletedTaskSpace
import parla

import logging
logging.basicConfig(level=logging.INFO)
# parla.tasks.logger.setLevel(logging.DEBUG)
# parla.task_runtime.logger.setLevel(logging.DEBUG)
# parla.cuda.logger.setLevel(logging.DEBUG)

# CPU code to perform a single step in the Jacobi iteration.
# Specialized later by jacobi_gpu
@specialized
@numba.njit(parallel=True)
def jacobi(a0, a1):
    a1[1:-1,1:-1] = .25 * (a0[2:,1:-1] + a0[:-2,1:-1] + a0[1:-1,2:] + a0[1:-1,:-2])


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
    threads_per_block = 64
    blocks_per_grid = (a0.shape[0] + (threads_per_block - 1)) // threads_per_block
    # Relying on numba/cupy interop here.
    gpu_jacobi_kernel[blocks_per_grid, threads_per_block](a0, a1)


def main():
    # Set up an "n" x "n" grid of values and run
    # "steps" number of iterations of the 4 point stencil on it.
    n = 4000
    steps = 6

    # Set up two arrays containing the input data.
    # This demo uses the standard technique of computing
    # from one array into another then swapping the
    # input and output arrays for the next iteration.
    # These are the two arrays that will be swapped back
    # and forth as input and output.
    a0 = np.random.rand(n, n)
    a1 = a0.copy()

    # Get the correct result using only the CPU to check
    # that the heterogeneous code is correct.
    actual = a0.copy()
    for i in range(steps):
        actual[1:-1,1:-1] = .25 * (actual[2:,1:-1] + actual[:-2,1:-1] + actual[1:-1,2:] + actual[1:-1,:-2])

    # Number of blocks in the partition across devices
    divisions = 40

    # An object that distributes arrays across all the given devices.
    mapper = LDeviceSequenceBlocked(divisions)

    # Partition a0 and a1.
    # Here we just partition the rows across the different devices.
    # Other partitioning schemes are possible.
    a0_row_groups = mapper.partition_tensor(a0, overlap=1)
    a1_row_groups = mapper.partition_tensor(a1, overlap=1)

    # Main parla task.
    @spawn(placement=cpu(0))
    async def run_jacobi():
        assert steps > 0
        # Specify which set of blocks is used as input or output
        # (they will be swapped for each iteration).
        in_blocks = a0_row_groups
        out_blocks = a1_row_groups
        # Create a set of labels for the tasks that perform the first
        # Jacobi iteration step.
        previous_block_tasks = CompletedTaskSpace()
        # Now create the tasks for subsequent iteration steps.
        for i in range(steps):
            # Swap input and output blocks for the next step.
            in_blocks, out_blocks = out_blocks, in_blocks
            # Create a new set of labels for the tasks that do this iteration step.
            current_block_tasks = TaskSpace("block_tasks[{}]".format(i))
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
                       dependencies=[previous_block_tasks[max(0, j-1):min(divisions, j+2)]],
                       placement=device)
                def device_local_jacobi_task():
                    # Read boundary values from adjacent blocks in the partition.
                    # This may communicate across device boundaries.
                    if j > 0:
                        copy(in_block[0], in_blocks[j - 1][-2])
                    if j < divisions - 1:
                        copy(in_block[-1], in_blocks[j + 1][1])
                    # Run the computation, dispatching to device specific code.
                    jacobi(in_block, out_block)
            # For the next iteration, use the newly created tasks as
            # the tasks from the previous step.
            previous_block_tasks = current_block_tasks
        await previous_block_tasks
        # Gather the results of the computation back into the original a1 array.
        # This depends on all the tasks from the last iteration step.
        for j in range(divisions):
            start_index = 1 if j > 0 else 0
            end_index = -1 if j < divisions - 1 else None  # None indicates the last element of the dimension
            copy(a1[mapper.slice(j, len(a1))], out_blocks[j][start_index:end_index])
    # Check that the heterogeneous computation matches
    # a very simple CPU only implementation.
    assert np.max(np.absolute(a1 - actual)) < 1E-14
    print("success")


if __name__ == '__main__':
    main()
