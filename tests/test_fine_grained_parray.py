from parla import Parla, parray

from parla.cuda import gpu
from parla.cpu import cpu
from parla.tasks import TaskSpace, spawn


import numpy as np


def main():
    @spawn(placement=cpu)
    async def test_blocked_cholesky():
        n = 2
        np.random.seed(10)
        # Construct input data
        a = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
        b = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
        a = parray.asarray(a)
        b = parray.asarray(b)
        print(a)

        ts = TaskSpace("CopyBack")

        @spawn(ts[0], placement=gpu[0], inout=[a[0]])
        def task1():
            a[0] += 10
            print(f"Task 1: {a[0].array}", flush=True)

        @spawn(ts[1], placement=gpu[0], inout=[a[1]])
        def task2():
            a[1] += 20
            print(f"Task 2: {a[1].array}", flush=True)

        @spawn(ts[2], placement=gpu[0], inout=[b[0]])
        def task3():
            b[0] += 30
            print(f"Task 3: {b[0].array}", flush=True)

        @spawn(ts[3], placement=gpu[1], inout=[b[1]])
        def task4():
            b[1] += 40
            print(f"Task 4: {b[1].array}", flush=True)

        @spawn(ts[4], [ts[0:2]], placement=gpu[1], output=[a])
        def task5():
            print(f"Task 5: {a}", flush=True)


        await ts
        print(a)
        

if __name__ == '__main__':
    with Parla():
        main()
