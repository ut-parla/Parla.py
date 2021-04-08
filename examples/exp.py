"""
Multi-device in-place exp computation
using parla with cupy as the kernel engine.
"""

import sys
import time

import numpy as np
import cupy as cp

from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import spawn, TaskSpace

def main():
    ngpus = int(sys.argv[1])
    runs = int(sys.argv[2])
    devices = gpu.devices[:ngpus]
    # 1D partition over available devices
    mapper = LDeviceSequenceBlocked(ngpus, placement = devices)

    # Generate an nxn array of random data and
    # partition it over the devices in use.
    n = 40000

    # Main task that generates others.
    @spawn(placement = cpu)
    async def rerun_exp():
        for run in range(runs):
            @spawn(placement = cpu)
            async def launch_exp():
                np.random.seed(0)
                a_cpu = np.random.rand(n, n).astype(np.float32)
                a_part = mapper.partition_tensor(a_cpu)
                print("launch")
                start = time.perf_counter()
                # A place to store tasks in order to refer
                # to them later for dependencies.
                exp_runs = TaskSpace("exp_runs")
                for i in range(ngpus):
                    # Launch a task for each GPU.
                    # These execute asynchronously.
                    @spawn(exp_runs[i], placement = mapper.device(i))
                    def run_exp():
                        # Call cupy for exponentiation.
                        # More complicated kernels can use numba.
                        local_start = time.perf_counter()
                        cp.exp(a_part[i], out = a_part[i])
                        cp.cuda.get_current_stream().synchronize()
                        local_stop = time.perf_counter()
                        print("local:", local_stop - local_start)
                # Wait for the exp tasks to complete
                # before measuring the end time.
                await exp_runs
                stop = time.perf_counter()
                print(stop - start)
            await launch_exp

if __name__ == "__main__":
    with Parla():
        main()
