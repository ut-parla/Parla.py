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

        for i in [0, 1]:
            if i != 0:
                @spawn(ts[1, i], [ts[5, i-1]], placement=gpu[1], inout=[b])
                def task11():
                    print(f"ts 1 {i}")
                    b[0, 0] += 10

                @spawn(ts[2, i], [ts[5, i-1]], placement=gpu[1], inout=[a])
                def task12():
                    print(f"ts 2 {i}")
                    a[0, 0] += 10
            else:
                @spawn(ts[1, i], placement=gpu, inout=[b])
                def task1():
                    print(f"ts 1 {i}")
                    b[0, 0] += 10

                @spawn(ts[2, i], placement=gpu, inout=[a])
                def task2():
                    print(f"ts 2 {i}")
                    a[0, 0] += 10

            @spawn(ts[3, i], [ts[2, i], ts[1, i]], placement=gpu[0], input=[a], inout=[b])
            def task3():
                print(f"ts 3 {i}")
                print(f"3: {a[0, 0]}")

            @spawn(ts[4, i], [ts[2, i], ts[1, i]], placement=gpu[0], input=[a])
            def task4():
                print(f"ts 4 {i}")
                print(f"4: {a[0, 0]}")

            @spawn(ts[5, i], [ts[3, i], ts[4, i]], placement=gpu[0], input=[b], inout=[a])
            def task5():
                print(f"ts 5 {i}")
                print(f"5: {b[0, 0]}")

        @spawn(ts[0, 0], [ts[5, i]], placement=gpu[1], input=[b], inout=[a])
        def task6():
            print(f"6.a.0: {a}")
            print(f"6.b.0: {b}")


            print(f"6.a: {a}")
            print(f"6.b: {b}")

            print(f"6.b.1: {b}")
        

if __name__ == '__main__':
    with Parla():
        main()
