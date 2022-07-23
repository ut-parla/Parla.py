"""
A simple inner product implemented in Parla.

This is probably the most basic example of Parla.
"""

from parla import Parla, TaskEnvironment
import numpy as np

from parla.array import copy, storage_size
from parla.cuda import gpu
from parla.cpu import cpu
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import *
import time
import os



def main():
    divisions = 10
    mapper = LDeviceSequenceBlocked(divisions)

    async def inner(a, b):
        a_part = mapper.partition_tensor(a)
        b_part = mapper.partition_tensor(b)
        # Create array to store partial sums from each logical device
        partial_sums = np.empty(len(a_part))
        # Define a space of task names for the product tasks
        P = TaskSpace("P")
        for i in range(len(a_part)):
            @spawn(P[i], data=[a_part[i], b_part[i]])
            def inner_local():
                # Perform the local inner product using the numpy multiply operation, @.
                partial_sums[i:i+1] = a_part[i] @ b_part[i]
        @spawn(dependencies=P, data=[partial_sums])
        def reduce():
            return np.sum(partial_sums)
        return await reduce


    @spawn()
    async def main_task():
        n = 3*1000
        a = np.random.rand(n)
        b = np.random.rand(n)
        print("Starting.", a.shape, b.shape)
        res = await inner(a, b)
        assert np.allclose(np.inner(a, b), res)
        print("Success.", res)


if __name__ == '__main__':
    # Start Parla runtime
    with Parla():
        main()
