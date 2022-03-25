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

def main():

    @spawn(placement=cpu)
    async def main_task():
        ngpus = cp.cuda.runtime.getDeviceCount()
        repetitions = int(sys.argv[1])

        # set up two n x n arrays to multiply together.
        # n is chosen so that all three can be
        # stored within the memory of a single GPU
        # so that strong scaling numbers make sense.
        n = 20000

        blocks = ngpus
        block_size = n // ngpus
        ordr = 'F'
        print("BlockSize: ", block_size, ngpus)

        # Overdecomposing doesn't actually seem to help in this case
        # with the current parla runtime. This may be related to
        # some weirdness within the scheduler though, so
        # we can leave the code for blocks in-place for further
        # testing later.

        np.random.seed(0)
        a_cpu = np.random.rand(n, n).astype(dtype=np.float32, order=ordr)
        b_cpu = np.random.rand(n, n).astype(dtype=np.float32, order=ordr)

        print("Finished Data Allocation", flush=True)
        # Partition the two arrays and set up the
        # partitioned array where the result will be stored.
        # This could also be done using a parla mapper object.

        a_part = []
        b_part = []
        c_part = []

        # Start all operans from CPU memory.
        for i in range(blocks):
            a_part.append(list())
            b_part.append(list())
            for j in range(blocks):
                si = slice(i*block_size: (i+1)*block_size)
                sj = slice(j*block_size: (+1)*block_size)
                a_part[i].append(a_cpu[si, sj])
                b_part[i].append(b_cpu[si, sj])

        for i in range(blocks):
            c_part.append(list())
            for j in range(blocks):
                c_part[i].append(np.empty((0, 0), dtype=np.float32, order=ordr))

        print(len(c_part), len(c_part[0]), c_part[0][0].shape)

        # 1. NEW: convert to parray in batch
        a_part = asarray_batch(a_part)
        b_part = asarray_batch(b_part)
        c_part = asarray_batch(c_part)

        print(len(c_part), len(c_part[0]))

        for repetition in range(repetitions):

            #reset cblocks to None
            for i in range(blocks):
                for j in range(blocks):
                    c_part[i][j].update(np.empty((0, 0), dtype=np.float32, order=ordr))

            matmul = TaskSpace("matmul")
            start = time.perf_counter()
            for i in range(blocks):
                for j in range(blocks):
                    a_block = a_part[i]
                    b_block = b_part[j]
                    c_block = c_part[i][j]

                    memsize = 2*block_size**2 + block_size*n*2

                    @spawn(matmul[i, j], placement = gpu(i%ngpus), memory=memsize, input=[a_block, b_block], output=[c_block])
                    def matmul_task():
                        a = a_block.array
                        b = b_block.array
                        c = c_block.array

                        stream = cp.cuda.get_current_stream()
                        stream.synchronize()

                        assert(a.device.id == b.device.id)
                        print(f"+({i}, {j}): ", a.shape, b.shape, c.shape, " | On Device: ", cp.cuda.runtime.getDevice(), a.device.id, flush=True)
                        local_start = time.perf_counter()
                        #perform the memory allocation inside the task on the device
                        c = a @ b.T
                        stream.synchronize()
                        local_end = time.perf_counter()

                        c_block.update(c)
                        c = c_block.array
                        print(f"-({i}, {j}): ", a.shape, b.shape, c.shape, " | Elapsed: ", local_end-local_start, flush=True)

            await matmul
            stop = time.perf_counter()
            print(f"Iteration {repetition} | Time elapsed: ", stop - start, flush=True)

if __name__ == "__main__":
    with Parla():
        main()
