import numpy as np
import cupy as cp

from parla import Parla
from parla.cuda import gpu
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace
from parla.array import clone_here

def gpu_sqrt(data, i):
    gpu_data = clone_here(data[i])
    print(f"{cp.cuda.get_current_stream()}: data[{i}] host to device transfer complete")
    cp.sqrt(gpu_data)
    print(f"{cp.cuda.get_current_stream()}: data[{i}] computation complete")
    data[i] = cp.asnumpy(gpu_data)
    print(f"{cp.cuda.get_current_stream()}: data[{i}] device to host transfer complete")

def main():
    print("Initializing data on host")
    data0 = np.random.rand(2**29) # 4 GB of double-precision random numbers
    data1 = np.random.rand(2**29) # 4 GB of double-precision random numbers
    data = [data0, data1]

    @spawn()
    async def task_launcher():

        print("\nStarting GPU tasks with no memory constraints")
        for i in range(2):
            @spawn(placement=gpu(0))
            async def t1():
                gpu_sqrt(data, i)

        await t1

        print("\nStarting GPU tasks with memory constraints")
        for i in range(2):
            @spawn(placement=gpu(0), memory=2**32)
            async def t2():
                gpu_sqrt(data, i)

if __name__ == '__main__':
    with Parla():
        main()
