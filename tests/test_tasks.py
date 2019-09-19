from time import sleep

import numpy as np
from pytest import skip

from parla.cpu import cpu
from parla.tasks import *


def repetitions():
    """Return an iterable of the repetitions to perform for probabilistic/racy tests."""
    return range(10)


def sleep_until(predicate):
    """Sleep until either `predicate()` is true or 2 seconds have passed."""
    for _ in range(10):
        if predicate():
            break
        sleep(0.2)
    assert predicate()


def test_spawn():
    task_results = []
    @spawn()
    def task():
        task_results.append(1)

    sleep_until(lambda: len(task_results) == 1)
    assert task_results == [1]


def test_spawn_await():
    task_results = []
    @spawn()
    async def task():
        task_results.append(1)
        @spawn()
        def subtask():
            task_results.append(2)
        await subtask
        task_results.append(3)

    sleep_until(lambda: len(task_results) == 3)
    assert task_results == [1, 2, 3]


def test_spawn_await_async():
    task_results = []
    @spawn()
    async def task():
        task_results.append(1)
        @spawn()
        async def subtask():
            await tasks()
            task_results.append(2)
        await subtask
        task_results.append(3)

    sleep_until(lambda: len(task_results) == 3)
    assert task_results == [1, 2, 3]


def test_await_value():
    task_results = []
    @spawn()
    async def task():
        @spawn()
        def subtask():
            return 42
        v = (await subtask)
        task_results.append(v)
        print(v)

    sleep_until(lambda: len(task_results) == 1)
    assert task_results == [42]


def test_await_value_async_source():
    task_results = []
    @spawn()
    async def task():
        @spawn()
        async def subtask():
            return 42
        task_results.append(await subtask)

    sleep_until(lambda: len(task_results) == 1)
    assert task_results == [42]


def test_spawn_await_id():
    task_results = []
    @spawn()
    async def task():
        task_results.append(1)
        B = TaskSpace()
        @spawn(B[0])
        def subtask():
            task_results.append(2)
        await B[0]
        task_results.append(3)

    sleep_until(lambda: len(task_results) == 3)
    assert task_results == [1, 2, 3]


def test_spawn_await_multi_id():
    task_results = []
    @spawn()
    async def task():
        task_results.append(1)
        B = TaskSpace()
        for i in range(10):
            @spawn(B[i])
            def subtask():
                task_results.append(2)
        await tasks(B[0:10])
        task_results.append(3)

    sleep_until(lambda: len(task_results) == 12)
    assert task_results == [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]


def test_dependencies():
    task_results = []
    @spawn()
    async def task():
        B = TaskSpace()
        C = TaskSpace()
        for i in range(10):
            @spawn(B[i], [C[i-1]] if i > 0 else [])
            def subtask():
                task_results.append(i)
            @spawn(C[i], [B[i]])
            def subtask():
                sleep(0.1) # Required delay to allow out of order execution without dependencies
                task_results.append(i+1)

    sleep_until(lambda: len(task_results) == 20)
    assert task_results == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10]


def test_closure_detachment():
    task_results = []
    @spawn()
    async def task():
        C = TaskSpace()
        for i in range(10):
            @spawn(C[i], [C[i-1]] if i > 0 else [])
            def subtask():
                sleep(0.1) # Required delay to allow out of order execution without dependencies
                task_results.append(i)

    sleep_until(lambda: len(task_results) == 10)
    assert task_results == list(range(10))


def test_fox_twice():
    loc = {
        (0, 0): cpu(0),
        (0, 1): cpu(0),
        (1, 0): cpu(0),
        (1, 1): cpu(0),
        }
    # Eventually: loc = infer_placements()
    # Or even better: loc = abstract_cartesian(n, n)
    def mem(i, j):
        return loc[(i,j)].memory()

    partitions_x = 2
    partitions_y = partitions_x

    def partition_slice(i, p, n):
        return slice(i * (n // p), (i + 1) * (n // p))


    def collect_fox(n, yp, y, done_task):
        C = TaskSpace()

        # reduce along rows
        for i in range(0, partitions_y):  # rows
            @spawn(C[i], [done_task], placement=cpu(0))
            def c():
                y[partition_slice(i, partitions_x, n)] = cpu(0).memory()(yp[i][i])

        # join the collect tasks
        @spawn(None, [C[0:partitions_y]], placement=cpu(0))
        def done():
            pass

        return done  # Return the join task


    def partition_fox(y, A, x):
        # n is the size of the arrays
        n = y.shape[-1]

        # check that inputs are the correct sizes
        assert y.shape == (n,)
        assert x.shape == (n,)
        assert A.shape == (n, n)

        assert n / partitions_x == n // partitions_x
        # partition A into Ap (partitions_x, partitions_y)
        Ap = [[mem(i, j)(A[partition_slice(i, partitions_x, n), partition_slice(j, partitions_y, n)])
               for j in range(partitions_x)]
              for i in range(partitions_y)]
        xp = [[mem(i, j)(x[partition_slice(j, partitions_x, n)]) if i == j else mem(i, j).np.empty(x[partition_slice(i, partitions_x, n)].shape)
               for j in range(partitions_x)]
              for i in range(partitions_y)]
        yp = [[mem(i, j).np.empty(y[partition_slice(i, partitions_x, n)].shape)
               for j in range(partitions_x)]
              for i in range(partitions_y)]

        return n, yp, Ap, xp


    def matvec_fox_partitioned(n, yp, Ap, xp):
        B = TaskSpace()
        M = TaskSpace()
        R = TaskSpace()

        # broadcast along columns
        for j in range(0, partitions_x): # columns
            for i in range(0, partitions_y): # rows
                @spawn(B[i, j], placement=loc[(i, j)])
                def b():
                    xp[i][j][:] = mem(i, j)(xp[j][j])

        # block-wise multiplication
        for i in range(0, partitions_y):  # rows
            for j in range(0, partitions_x): # columns
                @spawn(M[i, j], [B[i, j]], placement=loc[(i, j)])
                def m():
                    yp[i][j][:] = Ap[i][j] @ xp[i][j]

        # reduce along rows
        for i in range(0, partitions_y): # rows
            @spawn(R[i], [M[i, 0:partitions_x]], placement=loc[(i, i)])
            def r():
                acc = yp[i][i]
                # logger.info("acc = %r (at %r)", acc.device, get_current_device())
                for j in range(0, partitions_x): # columns
                    if i == j:
                        continue
                    t = mem(i, i)(yp[i][j])
                    # logger.info("%r, %r", t.device, yp[i][j].device)
                    acc[:] = acc + t

        # join the reduce tasks
        @spawn(None, [R[0:partitions_y]], placement=cpu(0))
        def done():
            pass
        return done  # Return the join task

    done_flag = []

    @spawn(placement=cpu(0))
    async def test_fox():
        size_factor = 4*partitions_x
        A = np.random.rand(size_factor, size_factor)
        x = np.random.rand(size_factor)
        # print_actual(A, x)
        res = A @ (A @ x)
        out = np.empty_like(x)
        n, yp, Ap, xp = partition_fox(out, A, x)
        T1 = matvec_fox_partitioned(n, yp, Ap, xp)
        await T1
        done = matvec_fox_partitioned(n, xp, Ap, yp)
        T = collect_fox(n, xp, out, done)
        await T
        assert np.allclose(res, out), "Parallel fox failed"
        done_flag.append(True)

    sleep_until(lambda: done_flag)
    assert done_flag


def test_placement():
    try:
        from parla.cuda import gpu
    except (ImportError, AttributeError):
        skip("CUDA required for this test.")

    for rep in repetitions():
        task_results = []
        for i in range(4):
            @spawn(placement=gpu(i))
            async def task():
                task_results.append(get_current_device())
            sleep_until(lambda: len(task_results) == i+1)

        assert task_results == [gpu(0), gpu(1), gpu(2), gpu(3)]


def test_placement_await():
    try:
        from parla.cuda import gpu
    except (ImportError, AttributeError):
        skip("CUDA required for this test.")

    for rep in repetitions():
        task_results = []
        for i in range(4):
            @spawn(placement=gpu(i))
            async def task():
                task_results.append(get_current_device())
                await tasks() # Await nothing to force a new task.
                task_results.append(get_current_device())
            sleep_until(lambda: len(task_results) == (i+1)*2)

        assert task_results == [gpu(0), gpu(0), gpu(1), gpu(1), gpu(2), gpu(2), gpu(3), gpu(3)]

