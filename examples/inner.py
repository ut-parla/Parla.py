"""
A simple inner product implemented in Parla.

This is probably the most basic example of Parla.
"""

from parla import Parla, TaskEnvironment
#from parla.multiload import MultiloadComponent, CPUAffinity

#with multiload():
#    import numpy as np
import numpy as np

from parla.array import copy, storage_size
from parla.cuda import gpu, GPUComponent
from parla.cpu import cpu, UnboundCPUComponent
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
                copy(partial_sums[i:i+1], a_part[i] @ b_part[i])
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
    # Setup task execution environments
    envs = []
    envs.extend([TaskEnvironment(placement=[d], components=[GPUComponent()]) for d in gpu.devices])
    envs.extend([TaskEnvironment(placement=[d], components=[UnboundCPUComponent()]) for d in cpu.devices])
    #envs.extend([TaskEnvironment(placement=[d], components=[MultiloadComponent([CPUAffinity])]) for d in cpu.devices])
    if "N_DEVICES" in os.environ:
        envs = envs[:int(os.environ.get("N_DEVICES"))]
    # Start Parla runtime
    with Parla(envs):
        main()
