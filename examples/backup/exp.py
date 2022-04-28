"""
Multi-device in-place exp computation
using parla with cupy as the kernel engine.
"""

import math
import sys
import time

import numpy as np

from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import spawn, TaskSpace

import cupy as cp
from numba import cuda, void, float32

@cuda.jit(void(float32[:]))
def inplace_exp(vals):
    thread_id = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(thread_id, vals.shape[0], stride):
        vals[i] = math.exp(vals[i])

def main():
    ngpus = int(sys.argv[1])
    runs = int(sys.argv[2])
    blocks_per_gpu = int(sys.argv[3])
    devices = gpu.devices[:ngpus]
    # 1D partition over available devices
    mapper = LDeviceSequenceBlocked(ngpus * blocks_per_gpu, placement = devices)

    # Generate an nxn array of random data and
    # partition it over the devices in use.
    n = 20000 * 20000

    # Main task that generates others.
    @spawn(placement = cpu)
    async def rerun_exp():
        for run in range(runs):
            @spawn(placement = cpu)
            async def launch_exp():
                np.random.seed(0)
                a_cpu = np.random.rand(n).astype(np.float32)
                #a_part = mapper.partition_tensor(a_cpu)
                a_part = []
                nblocks = ngpus * blocks_per_gpu
                block_size = (n - 1) // nblocks + 1
                for i in range(nblocks):
                    with cp.cuda.Device(i % ngpus):
                        a_part.append(cp.asarray(a_cpu[i * block_size : (i + 1) * block_size]))
                start = time.perf_counter()
                # A place to store tasks in order to refer
                # to them later for dependencies.
                exp_runs = TaskSpace("exp_runs")
                for i in range(ngpus * blocks_per_gpu):
                    # Launch a task for each GPU.
                    # These execute asynchronously.
                    @spawn(exp_runs[i], placement = a_part[i])
                    def run_exp():
                        # Call cupy for exponentiation.
                        # More complicated kernels can use numba.
                        #local_start = time.perf_counter()
                        cp.exp(a_part[i], out = a_part[i])
                        #a_loc = a_part[i]
                        #blocks = a_loc.shape[0] // (1024)
                        #threads_per_block = 512
                        #inplace_exp[blocks, threads_per_block](a_loc)
                        #cuda.default_stream().synchronize()
                        #cp.cuda.get_current_stream().synchronize()
                        #local_stop = time.perf_counter()
                        #print("local:", local_stop - local_start)
                # Wait for the exp tasks to complete
                # before measuring the end time.
                await exp_runs
                stop = time.perf_counter()
                print(stop - start)
            await launch_exp

if __name__ == "__main__":
    with Parla():
        main()
