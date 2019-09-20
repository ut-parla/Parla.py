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
            sleep(0.01)
            await tasks()
            sleep(0.01)
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


def test_placement():
    try:
        from parla.cuda import gpu
    except (ImportError, AttributeError):
        skip("CUDA required for this test.")

    devices = [cpu(0), gpu(0)]

    for rep in repetitions():
        task_results = []
        for i in range(2):
            @spawn(placement=devices[i])
            async def task():
                task_results.append(get_current_device())
            sleep_until(lambda: len(task_results) == i+1)

        assert task_results == devices


def test_placement_await():
    try:
        from parla.cuda import gpu
    except (ImportError, AttributeError):
        skip("CUDA required for this test.")

    devices = [cpu(0), gpu(0)]

    for rep in repetitions():
        task_results = []
        for i in range(2):
            @spawn(placement=devices[i])
            async def task():
                task_results.append(get_current_device())
                await tasks() # Await nothing to force a new task.
                task_results.append(get_current_device())
            sleep_until(lambda: len(task_results) == (i+1)*2)

        assert task_results == [cpu(0), cpu(0), gpu(0), gpu(0)]

