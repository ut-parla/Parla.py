import argparse
import os

parser = argparse.ArgumentParser()
#How many trials to run
parser.add_argument('-trials', type=int, default=1)
#What mapping to use
parser.add_argument('-fixed', type=int, default=1)
#How many gpus to use
parser.add_argument('-ngpus', type=int, default=1)
args = parser.parse_args()

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    print("CUDA_VISIBLE_DEVICES is not set. Assuming 0-3")
    cuda_visible_devices = list(range(4))
else:
    cuda_visible_devices = map(int, cuda_visible_devices.strip().split(','))

gpus = cuda_visible_devices[:args.gpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

#IMPORT CUPY AND PARLA AFTER CUDA_VISIBLE_DEVICES IS SET
import cupy as cp
import time
import numba.cuda
import numpy as np
import resource
import sys
from numba import cuda
from parla import Parla, get_all_devices
from parla.array import copy
from parla.cpu import cpu
from parla.cuda import gpu
from parla.function_decorators import specialized
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import spawn, TaskSpace, CompletedTaskSpace

from parla.parray import asarray, asarray_batch

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

    cp_stream = cp.cuda.get_current_stream()
    nb_stream = stream_cupy_to_numba(cp_stream)

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

        a0_row_groups = a0_row_groups._latest_view
        a1_row_groups = a1_row_groups._latest_view

        a0_row_groups = [ [a[0:1], a, a[-1:]] for a in a0_row_groups]
        a1_row_groups = [ [a[0:1], a, a[-1:]] for a in a1_row_groups]

        #a0_row_groups = [ [a0_row_groups[j][0][:]=j, a0_row_groups[j][1][:]=j, a0_row_groups[j][2][:=j]] for j in range(len(a0_row_groups))]
        #a1_row_groups = [ [a1_row_groups[j][0][:]=j, a1_row_groups[j][1][:]=j, a1_row_groups[j][2][:]=j] for j in range(len(a1_row_groups))]

        a0_row_groups = asarray_batch(a0_row_groups)
        a1_row_groups = asarray_batch(a1_row_groups)

        cs = TaskSpace("Compilation Space")
        for i in range(divisions):
            @spawn(cs[i], placement=mapper.device(i))
            def w():
                print("Compile: ", i, flush=True)
                jacobi(a1_row_groups[i][1].array, a0_row_groups[i][1].array)

        await cs

        for i in range(num_tests):

            if i > 0:
                rs = TaskSpace("Reset")
                for k in range(divisions):
                    for l in range(3):
                        @spawn(rs[k, l], placement=mapper.device(k), inout=[a1_row_groups[k][l], a0_row_groups[k][l]])
                        def reset():
                            noop = 1
                            #print("Reseting Parrays to Modified")
                await rs


            # Specify which set of blocks is used as input or output
            # (they will be swapped for each iteration).

            in_blocks = a0_row_groups
            out_blocks = a1_row_groups

            # Create a set of labels for the tasks that perform the first
            # Jacobi iteration step.

            previous_block_tasks = CompletedTaskSpace()

            # Now create the tasks for subsequent iteration steps.

            start = time.perf_counter()
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

                    if args.fixed:
                        device = mapper.device(j)
                    else:
                        device = devs

                    in_list = list()
                    if j > 0:
                        in_list.append(in_blocks[j-1][2])
                    if j < divisions-1:
                        in_list.append(in_blocks[j+1][0])

                    out_list = [ in_blocks[j][1], out_blocks[j][1], out_blocks[j][0], out_blocks[j][2] ]

                    #print("spawning", flush=True)

                    # Make each task operating on each block depend on the tasks for
                    # that block and its immediate neighbors from the previous iteration.

                    @spawn(current_block_tasks[j],
                           dependencies=[previous_block_tasks[max(0, j-1):min(divisions, j+2)]],
                           input=in_list,
                           inout=out_list,
                           placement=device)
                    def device_local_jacobi_task():

                        # Read boundary values from adjacent blocks in the partition.
                        # This may communicate across device boundaries.
                        #if j > 0:
                        #    copy(in_block[0], in_blocks[j - 1][-2])
                        #if j < divisions - 1:
                        #    copy(in_block[-1], in_blocks[j + 1][1])

                        # Run the computation, dispatching to device specific code.
                        #jacobi(in_block, out_block)
                        #print("+Jacobi Task", i, j, flush=True)
                        #print("=Jacobi Task", i, j, "Copy Shapes: ", in_blocks[j][0].shape, in_blocks[j][1].shape, in_blocks[j][2].shape, in_blocks[j][1].device.id, flush=True)
                        stream = cp.cuda.get_current_stream()

                        data_start = time.perf_counter()
                        if j > 0:
                            #print("=Jacobi Task", i, j, "Shape: ", in_blocks[j][1].array[0].shape, in_blocks[j-1][2].array[-1].shape, flush=True)
                            in_blocks[j][1].array[0] =  in_blocks[j-1][2].array[-1]
                        if j < divisions - 1:
                            #print("=Jacobi Task", i, j,"Shape: ", in_blocks[j][1].array[-1].shape, in_blocks[j+1][0].array[0].shape, flush=True)
                            in_blocks[j][1].array[-1] =  in_blocks[j+1][0].array[0]
                        #stream.synchronize()
                        data_end = time.perf_counter()

                        #print("Data: ", data_end - data_start, flush=True)

                        # Run the computation, dispatching to device specific code.

                        #print(in_blocks[j][1].array.shape, out_blocks[j][1].array.shape)
                        compute_start = time.perf_counter()
                        jacobi(in_blocks[j][1].array, out_blocks[j][1].array)
                        stream.synchronize()
                        compute_end = time.perf_counter()
                        #print("Compute: ", compute_end - compute_start, flush=True)

                        #print("=Jacobi Task", i, j, "Interior Shape: ", in_blocks[j][1].array.shape, out_blocks[j][1].shape, flush=True)

                        #Copy over to new output
                        data_start = time.perf_counter()
                        out_blocks[j][0].array[:] = out_blocks[j][1].array[:1]
                        out_blocks[j][2].array[:] = out_blocks[j][1].array[-1:]
                        #stream.synchronize()
                        data_end = time.perf_counter()

                        #print("Copy: ", data_end - data_start, flush=True)

                        #Copy over to new output
                        #out_blocks[j][0].update(out_blocks[j][1].array[:1])
                        #out_blocks[j][2].update(out_blocks[j][1].array[-1:])
                        #print("-Jacobi Task", i, flush=True)

                # For the next iteration, use the newly created tasks as
                # the tasks from the previous step.
                previous_block_tasks = current_block_tasks

            #end_spawn = time.perf_counter()
            #print("Time to Spawn: ", end_spawn - start_spawn)

            await current_block_tasks
            end = time.perf_counter()
            print("Total Time: ", end - start, "seconds")

        del a0
        del a1
        del a0_row_groups
        del a1_row_groups
        del mapper

if __name__ == '__main__':
    with Parla():
        main()
