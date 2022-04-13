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
        b = parray.asarray(a)
        print(a)

        ts = TaskSpace("CopyBack")

        for i in [0]:
            if i != 0:
                @spawn(ts[1+i, 1], [ts[4+(i-10), 1]], placement=gpu, inout=[b])
                def task2():
                    b[0, 0] += 10

                @spawn(ts[2+i, 1], [ts[4+(i-10), 1]], placement=gpu, inout=[a])
                def task3():
                    a[0, 0] += 10
            else:
                @spawn(ts[1+i, 1], placement=gpu, inout=[b])
                def task2():
                    b[0, 0] += 10

                @spawn(ts[2+i, 1], placement=gpu, inout=[a])
                def task3():
                    a[0, 0] += 10

            @spawn(ts[3+i, 1], [ts[2+i, 1], ts[1+i, 1]], placement=gpu, input=[a], inout=[b])
            def task4():
                print(f"4: {a[0, 0]}")

            @spawn(ts[5+i, 1], [ts[2+i, 1], ts[1+i, 1]], placement=gpu, input=[a])
            def task6():
                print(f"6: {a[0, 0]}")

            @spawn(ts[4+i, 1], [ts[3+i, 1]], placement=gpu[0], input=[b], inout=[a])
            def task5():
                print(f"5: {b[0, 0]}")

        @spawn(ts[0, 0], [ts[4, 1]], placement=gpu[1], input=[b], inout=[a])
        def task6():
            print(f"6.a.0: {a}")
            print(f"6.b.0: {b}")

            b.evict()
            a.evict()

            print(f"6.a: {a}")
            print(f"6.b: {b}")

            b.evict(0, keep_one_copy=False)
            print(f"6.b.1: {b}")
        

if __name__ == '__main__':
    with Parla():
        main()
