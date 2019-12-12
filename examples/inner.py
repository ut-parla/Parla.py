"""
A simple inner product implemented in Parla.

This is probably the most basic example of Parla.
"""
import logging

import numpy as np

from parla import task_runtime
from parla.array import copy
from parla.cuda import gpu
from parla.cpu import cpu
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import *
import parla
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
parla.tasks.logger.setLevel(logging.DEBUG)
# parla.array.logger.setLevel(logging.DEBUG)
# parla.cuda.logger.setLevel(logging.DEBUG)
# parla._cpuutils.logger.setLevel(logging.DEBUG)


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

    @spawn(resources=[{cpu: 1}])
    async def inner_part():
        # Create array to store partial sums from each logical device
        partial_sums = np.empty(divisions)

        # Start a block of tasks that much all complete before leaving the block.
        async with finish():
            # For each logical device, perform the local inner product using the numpy multiply operation, @.
            for i in range(divisions):
                @spawn(resources=[{gpu: 1024}, {cpu: 1}],
                       reads=[a_part[i], b_part[i]], writes=[partial_sums[i:i+1]],
                       cost=lambda rs: len(a_part[i]) * (1 if gpu.threads in rs else 2),
                       # constraints=lambda assignment: assignment == mapper.device(i))
                       )
                def inner_local():
                    print("Compute", i)
                    copy(partial_sums[i:i+1], a_part[i] @ b_part[i])
        # Reduce the partial results (sequentially)
        print("Reduce")
        res = 0.
        for i in range(divisions):
            res += partial_sums[i]
        inner_result[0] = res

    @spawn(None, [inner_part],
           resources=[{cpu: 1}])
    def check():
        print("Check")
        end = time.perf_counter()
        print(end - start)

        assert np.allclose(np.inner(a, b), inner_result[0])


if __name__ == '__main__':
    with task_runtime.Scheduler(16):
        main()
