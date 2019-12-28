"""
A simple inner product implemented in Parla.

This is probably the most basic example of Parla.
"""
import numpy as np

from parla import Parla
from parla.array import copy, storage_size
from parla.cuda import gpu
from parla.cpu import cpu
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import *
import time
import os

def main():
    n = 3*1000
    a = np.random.rand(n)
    b = np.random.rand(n)

    divisions = 10

    start = time.perf_counter()
    # Map the divisions onto actual hardware locations
    devs = list(gpu.devices) + list(cpu.devices)
    # devs = cpu.devices
    if "N_DEVICES" in os.environ:
        devs = devs[:int(os.environ.get("N_DEVICES"))]
    mapper = LDeviceSequenceBlocked(divisions, devices=devs)

    a_part = mapper.partition_tensor(a)
    b_part = mapper.partition_tensor(b)

    inner_result = np.empty(1)

    @spawn()
    async def inner_part():
        # Create array to store partial sums from each logical device
        partial_sums = np.empty(divisions)
        # Start a block of tasks that much all complete before leaving the block.
        async with finish():
            # For each logical device, perform the local inner product using the numpy multiply operation, @.
            for i in range(divisions):
                @spawn(devices=[Req(mapper.device(i), threads=1, memory=storage_size(a_part[i], b_part[i]))])
                def inner_local():
                    copy(partial_sums[i:i+1], a_part[i] @ b_part[i])
        # Reduce the partial results (sequentially)
        res = 0.
        for i in range(divisions):
            res += partial_sums[i]
        inner_result[0] = res

    @spawn(None, [inner_part])
    def check():
        end = time.perf_counter()
        print(end - start)

        assert np.allclose(np.inner(a, b), inner_result[0])


if __name__ == '__main__':
    with Parla():
        main()
