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
from parla.tasks import spawn, TaskSpace, CompletedTaskSpace

def main():
    ngpus = int(sys.argv[1])
    repetitions = int(sys.argv[2])

    # set up two n x n arrays to multiply together.
    # n is chosen so that all three can be
    # stored within the memory of a single GPU
    # so that strong scaling numbers make sense.
    n = 20000
    blocks = ngpus * 2
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
    for i in range(blocks):
        with cp.cuda.Device(i % ngpus):
            a_part.append(cp.array(a_cpu[i * block_size : (i + 1) * block_size], order = 'F'))
            b_part.append(cp.array(b_cpu[i * block_size : (i + 1) * block_size], order = 'F'))
            c_dim = b_part[-1].shape[0]
            c_part.append(cp.empty((c_dim, n), np.float32, order = 'F'))

    previous = None
    for repetition in range(repetitions):
        # Now compute a @ b.T and write the output to c
        deps = [previous] if previous is not None else []
        @spawn(placement = cpu, dependencies = deps)
        async def run_matmul():
            print("start")
            start = time.perf_counter()
            matmul = TaskSpace("matmul")
            for i in range(blocks):
                for j in range(blocks):
                    a_block = a_part[i]
                    b_block = b_part[j]
                    c_block = c_part[i][:, j * block_size : (j + 1) * block_size]
                    @spawn(matmul[i, j], placement = c_block)
                    def matmul_task():
                        old_device = cp.cuda.Device()
                        #b_block_local = clone_here(b_block)
                        b_block_local = cp.asarray(b_block, order = 'F')
                        # cupy doesn't support the out argument for matmul yet so we have to copy.
                        # cp.matmul(a_block, b_block_local.T, out = c_block)
                        #print(i, j, old_device, cp.cuda.Device(), c_block.device, a_block.device, b_block_local.device, b_block.device)
                        c_block[:] = a_block @ b_block_local.T
            await matmul
            stop = time.perf_counter()
            print(stop - start)
        previous = run_matmul

if __name__ == "__main__":
    with Parla():
        main()
