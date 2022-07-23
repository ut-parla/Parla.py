import time
import numba.cuda
import numpy as np
import cupy as cp
from numba import cuda

import resource
import sys
from parla import Parla, get_all_devices
from parla.array import copy
from parla.cpu import cpu
from parla.cuda import gpu
from parla.function_decorators import specialized
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import spawn, TaskSpace, CompletedTaskSpace
import argparse

parser = argparse.ArgumentParser()
#How many trials to run
parser.add_argument('-trials', type=int, default=1)
#What mapping to use
parser.add_argument('-fixed', type=int, default=1)
args = parser.parse_args()

num_tests = args.trials

def stream_cupy_to_numba(cp_stream):
    '''
    Notes:
        1. The lifetime of the returned Numba stream should be as long as the CuPy one,
           which handles the deallocation of the underlying CUDA stream.
        2. The returned Numba stream is assumed to live in the same CUDA context as the
           CuPy one.
        3. The implementation here closely follows that of cuda.stream() in Numba.
    '''
    from ctypes import c_void_p
    import weakref

    # get the pointer to actual CUDA stream
    raw_str = cp_stream.ptr

    # gather necessary ingredients
    ctx = cuda.devices.get_context()
    handle = c_void_p(raw_str)
    finalizer = None  # let CuPy handle its lifetime, not Numba

    # create a Numba stream
    nb_stream = cuda.cudadrv.driver.Stream(weakref.proxy(ctx), handle, finalizer)

    return nb_stream

def using(point=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return "%s: usertime=%s systime=%s mem=%s mb \
           "%(point, usage[0], usage[1],
                usage[2] / 1024.0 )
@specialized
@numba.njit(parallel=True)
def jacobi(a0, a1):
    """
    CPU code to perform a single step in the Jacobi iteration.
    """
    a1[1:-1,1:-1] = .25 * (a0[2:,1:-1] + a0[:-2,1:-1] + a0[1:-1,2:] + a0[1:-1,:-2])
@jacobi.variant(gpu)
def jacobi_gpu(a0, a1):
    """
    GPU kernel call to perform a single step in the Jacobi iteration.
    """
    threads_per_block_x = 32
    threads_per_block_y = 1024//threads_per_block_x
    blocks_per_grid_x = (a0.shape[0] + (threads_per_block_x - 1)) // threads_per_block_x
    blocks_per_grid_y = (a0.shape[1] + (threads_per_block_y - 1)) // threads_per_block_y

    nb_stream = stream_cupy_to_numba(cp.cuda.get_current_stream())
    gpu_jacobi_kernel[(blocks_per_grid_x,blocks_per_grid_y), (threads_per_block_x,threads_per_block_y), nb_stream](a0, a1)
    nb_stream.synchronize()


@numba.cuda.jit
def gpu_jacobi_kernel(a0, a1):
    """
    Actual CUDA kernel to do a single step.
    """
    i, j = numba.cuda.grid(2)
    if 0 < i < a1.shape[0]-1 and 0 < j < a1.shape[1]-1:
        a1[i,j] = .25 * (a0[i-1,j] + a0[i+1,j] + a0[i,j-1] + a0[i,j+1])


def main():


    #divisions = len(get_all_devices())*2
    devs = gpu.devices
    divisions = len(devs)
    # Set up an "n" x "n" grid of values and run
    # "steps" iterations of the 4 point stencil on it.
    n = 30000
    steps = 500
    #steps = 1

    @spawn(placement=cpu)
    async def main_jacobi():
        # Set up two arrays containing the input data.
        # This demo uses the standard technique of computing
        # from one array into another then swapping the
        # input and output arrays for the next iteration.
        # These are the two arrays that will be swapped back
        # and forth as input and output.

        a0 = np.random.rand(n, n)
        a1 = a0.copy()
        # An object that distributes arrays across all the given devices.
        mapper = LDeviceSequenceBlocked(divisions, placement=devs)
        # Partition a0 and a1.
        # Here we just partition the rows across the different devices.
        a0_row_groups = mapper.partition_tensor(a0, overlap=1)
        a1_row_groups = mapper.partition_tensor(a1, overlap=1)

        #print([a.shape for a in a0_row_groups._latest_view])

        warmup = TaskSpace()
        for i in range(divisions):
            @spawn(warmup[i], placement=mapper.device(i))
            def w():
                jacobi(a1_row_groups[i], a0_row_groups[i])
        await warmup

        for k in range(num_tests):

            # Specify which set of blocks is used as input or output
            # (they will be swapped for each iteration).

            in_blocks = a0_row_groups.base
            out_blocks = a1_row_groups.base

            # Create a set of labels for the tasks that perform the first
            # Jacobi iteration step.
            tslist = list()
            previous_block_tasks = CompletedTaskSpace()

            start = time.perf_counter()

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

                    if args.fixed:
                        device = mapper.device(j)
                    else:
                        device = devs

                    #print("Placement: ", device)

                    in_block = in_blocks[j]
                    out_block = out_blocks[j]

                    # Make each task operating on each block depend on the tasks for
                    # that block and its immediate neighbors from the previous iteration.
                    @spawn(current_block_tasks[j],
                           dependencies=[previous_block_tasks[max(0, j-1):min(divisions, j+2)]],
                           placement=device)
                    def device_local_jacobi_task():
                        #print("+Jacobi Task", i, j, flush=True)
                        # Read boundary values from adjacent blocks in the partition.
                        # This may communicate across device boundaries.

                        stream = cp.cuda.get_current_stream()
                        data_start = time.perf_counter()
                        if j > 0:
                            #print("=Jacobi Task", i, j, "Shape: ", in_block[0].shape, in_blocks[j-1][-2].shape, flush=True)
                            copy(in_block[0], in_blocks[j-1][-1])
                        if j < divisions - 1:
                            #print("=Jacobi Task", i, j, "Shape: ", in_block[-1].shape, in_blocks[j+1][1].shape, flush=True)
                            copy(in_block[-1], in_blocks[j+1][0])
                        data_end = time.perf_counter()
                        #stream.synchronize()
                        #print("Data: ", data_end - data_start, flush=True)

                        # Run the computation, dispatching to device specific code.

                        #print(in_block.shape, out_block.shape, flush=True)
                        compute_start = time.perf_counter()
                        jacobi(in_block, out_block)
                        stream.synchronize()
                        compute_end = time.perf_counter()
                        #print("Compute: ", compute_end - compute_start, flush=True)
                        #print("=Jacobi Task", i, j, "Interior Shape: ", in_block.shape, out_block.shape, flush=True)
                        #print("-Jacobi Task", i, j, flush=True)

                # For the next iteration, use the newly created tasks as
                # the tasks from the previous step.

                tslist.append(previous_block_tasks)
                previous_block_tasks = current_block_tasks

            await current_block_tasks
            end = time.perf_counter()
            print("Time: ", end - start, "seconds")

            # This depends on all the tasks from the last iteration step.
            #for j in range(divisions):
            #    start_index = 1 if j > 0 else 0
            #    end_index = -1 if j < divisions - 1 else None  # None indicates the last element of the dimension
            #    copy(a1[mapper.slice(j, len(a1))], out_blocks[j][start_index:end_index])

        del a0
        del a1
        del a0_row_groups
        del a1_row_groups
        del mapper

if __name__ == '__main__':
    with Parla():
        main()
