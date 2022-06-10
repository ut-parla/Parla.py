"""
Multi-device matrix multiplication using parla with cupy as the kernel engine.

"""
import sys
import time

import numpy as np
import cupy as cp

from parla import Parla, get_all_devices
from parla.array import copy, clone_here
from parla.cpu import cpu
from parla.cuda import gpu
from parla.function_decorators import specialized
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import spawn, TaskSpace, CompletedTaskSpace, reserve_persistent_memory
from parla.parray import asarray_batch
import argparse

parser = argparse.ArgumentParser()
#How many trials to run
parser.add_argument('-trials', type=int, default=1)
#Are the placement fixed by the user or determed by the scheduler?
parser.add_argument('-fixed', default=0, type=int)
args = parser.parse_args()

def main():

    @spawn(placement=cpu)
    async def main_task():
        ngpus = cp.cuda.runtime.getDeviceCount()
        repetitions = args.trials

        # set up two n x n arrays to multiply together.
        # n is chosen so that all three can be
        # stored within the memory of a single GPU
        # so that strong scaling numbers make sense.
        n = 32000

        blocks = ngpus
        block_size = n // ngpus
        h_ordr = 'C'
        d_ordr = 'F'
        print("BlockSize: ", block_size, "GPUS: ", ngpus)

        # Overdecomposing doesn't actually seem to help in this case
        # with the current parla runtime. This may be related to
        # some weirdness within the scheduler though, so
        # we can leave the code for blocks in-place for further
        # testing later.

        np.random.seed(0)
        a_cpu = np.random.rand(n, n).astype(dtype=np.float32, order=h_ordr)
        b_cpu = np.random.rand(n, n).astype(dtype=np.float32, order=h_ordr)

        print("Finished Data Allocation", flush=True)
        # Partition the two arrays and set up the
        # partitioned array where the result will be stored.
        # This could also be done using a parla mapper object.

        a_part = []
        b_part = []
        c_part = []

        distribute=True
        reset=True
        fixed_placement=args.fixed
        verbose=False
        sync=False

        time_list = list()

        # Start all operans from CPU memory.
        for i in range(blocks):
            if distribute:
                with cp.cuda.Device(i):
                    a_part.append(cp.asarray(a_cpu[i * block_size : (i + 1) * block_size], order=d_ordr))
                    b_part.append(cp.asarray(b_cpu[i * block_size : (i + 1) * block_size], order=d_ordr))
                    cp.cuda.stream.get_current_stream().synchronize()
            else:
                a_part.append(a_cpu[i * block_size : (i + 1) * block_size])
                b_part.append(b_cpu[i * block_size : (i + 1) * block_size])

        for i in range(blocks):
            c_part.append(list())
            for j in range(blocks):
                c_part[i].append(np.empty((0, 0), dtype=np.float32, order=h_ordr))

        #print(len(c_part), len(c_part[0]), c_part[0][0].shape)

        # 1. NEW: convert to parray in batch
        a_part, b_part = asarray_batch(a_part, b_part)
        c_part = asarray_batch(c_part)

        #print(len(c_part), len(c_part[0]))


        for repetition in range(repetitions):

            #reset cblocks to None
            for i in range(blocks):
                for j in range(blocks):
                    c_part[i][j].update(np.empty((0, 0), dtype=np.float32, order=h_ordr))

            if reset:
                #reset coherence to only be in starting locations
                rspace = TaskSpace("reset")
                for i in range(blocks):
                    @spawn(rspace[i], placement=gpu(i%ngpus), memory=2*block_size*n, inout=[a_part[i], b_part[i]])
                    def reset_task():
                        a_part[i].update(a_part[i].array)
                        b_part[i].update(b_part[i].array)
                await rspace

            matmul = TaskSpace("matmul")
            start = time.perf_counter()
            for i in range(blocks):
                for j in range(blocks):
                    a_block = a_part[i]
                    b_block = b_part[j]
                    c_block = c_part[i][j]

                    memsize = (block_size**2)*4

                    if fixed_placement:
                        loc = gpu(i%ngpus)
                    else:
                        loc = gpu

                    @spawn(matmul[i, j], placement = loc, memory=memsize, input=[a_block, b_block], output=[c_block])
                    def matmul_task():
                        a = a_block.array
                        b = b_block.array
                        c = c_block.array

                        stream = cp.cuda.get_current_stream()
                        stream.synchronize()

                        assert(a.device.id == b.device.id)
                        if verbose:
                            print(f"+({i}, {j}): ", a.shape, b.shape, c.shape, " | On Device: ", cp.cuda.runtime.getDevice(), a.device.id, flush=True)
                        local_start = time.perf_counter()
                        c = a @ b.T

                        if sync:
                            stream.synchronize()
                        local_end = time.perf_counter()

                        c_block.update(c)
                        c = c_block.array

                        if verbose:
                            print(f"-({i}, {j}): ", a.shape, b.shape, c.shape, " | Elapsed: ", local_end-local_start, flush=True)

            await matmul
            stop = time.perf_counter()
            print(f"Iteration {repetition} | Time elapsed: ", stop - start, flush=True)
            time_list.append(stop-start)

        mean = np.mean(np.array(time_list))
        median = np.median(np.array(time_list))

        print(f"Execution Time:: Average = {mean} | Median = {median}", flush=True)

if __name__ == "__main__":
    with Parla():
        main()
