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
    ngpus = int(sys.argv[1])
    repetitions = int(sys.argv[2])

    # set up two n x n arrays to multiply together.
    # n is chosen so that all three can be
    # stored within the memory of a single GPU
    # so that strong scaling numbers make sense.
    n = 20000
    # Overdecomposing doesn't actually seem to help in this case
    # with the current parla runtime. This may be related to
    # some weirdness within the scheduler though, so
    # we can leave the code for blocks in-place for further
    # testing later.
    blocks = ngpus
    np.random.seed(0)
    a_cpu = np.random.rand(n, n).astype(np.float32, order = 'F')
    b_cpu = np.random.rand(n, n).astype(np.float32, order = 'F')
    # Partition the two arrays and set up the
    # partitioned array where the result will be stored.
    # This could also be done using a parla mapper object.
    a_part = []
    b_part = []
    c_part = []
    block_size = n // ngpus + 1
    # Start all operans from CPU memory.
    for i in range(blocks):
        a_part.append(a_cpu[i * block_size : (i + 1) * block_size])
        b_part.append(b_cpu[i * block_size : (i + 1) * block_size])
        c_dim = b_part[-1].shape[0]
        c_part.append(np.empty((c_dim, n), np.float32, order = 'F'))

    # 1. NEW: convert to parray in batch
    a_part, b_part, c_part = asarray_batch(a_part, b_part, c_part)

    previous = None
    matmul = TaskSpace("matmul")
    outer = TaskSpace("outer")
    for repetition in range(repetitions):
        offset = blocks * repetition
        # Now compute a @ b.T and write the output to c
        deps = [previous, matmul[offset-1, blocks-1]] if previous is not None else []
        # 2. NEW: input/inout
        @spawn(outer[repetition], placement = cpu, dependencies = deps, input=[*a_part, *b_part], inout=[*c_part])
        async def run_matmul():
            start = time.perf_counter()
            for i in range(blocks):
                for j in range(blocks):
                    a_block = a_part[i]
                    b_block = b_part[j]
                    c_block = c_part[i][:, j * block_size : (j + 1) * block_size]
                    memsize = c_block.nbytes
                    if i != j:
                        memsize += b_block.nbytes

                    k = i + offset
                    # 3. NEW: input/output
                    @spawn(matmul[k, j], dependencies=[matmul[0:k, j], matmul[k, 0:j]], placement = gpu, memory=memsize, input=[a_block, b_block], output=[c_block])
                    def matmul_task():
                        c_block[:] = (a_block @ b_block.T).array
                        # TODO(lhc): For now, do not copy back to cpu memory.
                        #            This is because I don't know how to move back to
                        #            PArray to CPU on nested task case.
                        #            Correctness check is done by printing c blocks.
            await matmul
            stop = time.perf_counter()
            print(stop - start)
        previous = run_matmul

if __name__ == "__main__":
    with Parla():
        main()
