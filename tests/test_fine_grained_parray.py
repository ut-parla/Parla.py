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

        print(ts[0:1, 0:2])

        @spawn(ts[0], placement=gpu[0], inout=[a[0]])
        def task2():
            a[0] += 10
            print(a[0])

        # @spawn(ts[1], placement=gpu, input=[a[1]])
        # def task3():
        #     a[1] += 10

        @spawn(ts[2], placement=gpu[1], inout=[b[0]])
        def task2():
            b[0] += 10
            print(b[0])

        # @spawn(ts[3], placement=gpu, input=[b[1]])
        # def task3():
        #     a[1] += 10

        

if __name__ == '__main__':
    with Parla():
        main()
