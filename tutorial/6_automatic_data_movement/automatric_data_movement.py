import numpy as np
import cupy as cp

from parla import Parla, parray
from parla.cuda import gpu
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace

def main():
    print("Initializing PArray")
    A = parray.asarray([[1, 2], [3, 4]])
    data = np.random.rand(2,2)
    B = parray.asarray(data)

    @spawn(placement=cpu)
    async def task_launcher():
        ts = TaskSpace("tasks")

        @spawn(ts[0], placement=gpu(0), input=[A])
        async def read_only_task():
            print(A)

        @spawn(ts[1], placement=gpu(1), output=[B])
        async def write_only_task():
            B.update(cp.sqrt(B.array))

        @spawn(ts[2], [ts[0:2]], placement=cpu, input=[A], inout=[B])
        async def write_and_write_task():
            B[0][1] = np.sum((B + A).array)

        @spawn(ts[3], [ts[2]], placement=gpu(1), inout=[A[0]])
        async def write_and_write_task():
            A[0] = cp.sqrt(A[0].array)

        await ts

if __name__ == '__main__':
    with Parla():
        main()