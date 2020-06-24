import logging
from itertools import combinations
from time import sleep

import numpy as np
import pytest
from pytest import skip

from parla import Parla, array, TaskEnvironment
from parla.cpu import cpu
from parla.tasks import *

logger = logging.getLogger(__name__)

def repetitions():
    """Return an iterable of the repetitions to perform for probabilistic/racy tests."""
    return range(10)


def sleep_until(predicate, timeout=2, period=0.05):
    """Sleep until either `predicate()` is true or 2 seconds have passed."""
    for _ in range(int(timeout/period)):
        if predicate():
            break
        sleep(period)
    assert predicate(), "sleep_until timed out ({}s)".format(timeout)


def test_parla_ctx():
    # Dummy environments with no components for testing.
    environments = [TaskEnvironment(placement=[d], components=[]) for d in cpu.devices]
    with Parla(environments):
        task_results = []
        @spawn()
        def task():
            task_results.append(1)

        sleep_until(lambda: len(task_results) == 1)
        assert task_results == [1]


def test_spawn(runtime_sched):
    task_results = []
    @spawn()
    def task():
        task_results.append(1)

    sleep_until(lambda: len(task_results) == 1)
    assert task_results == [1]


def test_spawn_await(runtime_sched):
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


def test_spawn_await_async(runtime_sched):
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


def test_await_value(runtime_sched):
    task_results = []
    @spawn()
    async def task():
        @spawn()
        def subtask():
            return 42
        v = (await subtask)
        task_results.append(v)

    sleep_until(lambda: len(task_results) == 1)
    assert task_results == [42]


def test_await_value_async_source(runtime_sched):
    task_results = []
    @spawn()
    async def task():
        @spawn()
        async def subtask():
            return 42
        task_results.append(await subtask)

    sleep_until(lambda: len(task_results) == 1)
    assert task_results == [42]


def test_spawn_await_id(runtime_sched):
    task_results = []
    @spawn()
    async def task():
        task_results.append(1)
        B = TaskSpace()
        @spawn(B[0], )
        def subtask():
            task_results.append(2)
        await B[0]
        task_results.append(3)

    sleep_until(lambda: len(task_results) == 3)
    assert task_results == [1, 2, 3]


def test_spawn_await_multi_id(runtime_sched):
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


def test_finish(runtime_sched):
    task_results = []
    @spawn()
    async def task():
        task_results.append(1)
        async with finish():
            for i in range(10):
                @spawn()
                def subtask():
                    sleep(0.05)
                    task_results.append(2)
        task_results.append(3)

    sleep_until(lambda: len(task_results) == 12)
    assert task_results == [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]


def test_finish_nested(runtime_sched):
    task_results = []
    @spawn()
    async def task():
        task_results.append(1)
        async with finish():
            for i in range(3):
                @spawn()
                async def subtask():
                    @spawn()
                    def subsubtask():
                        sleep(0.4)
                        task_results.append(2)
                    await subsubtask
        task_results.append(3)

    sleep_until(lambda: len(task_results) == 5)
    assert task_results == [1, 2, 2, 2, 3]


def test_dependencies(runtime_sched):
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
                sleep(0.05) # Required delay to allow out of order execution without dependencies
                task_results.append(i+1)

    sleep_until(lambda: len(task_results) == 20)
    assert task_results == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10]


def test_closure_detachment(runtime_sched):
    task_results = []
    @spawn()
    async def task():
        C = TaskSpace()
        for i in range(10):
            @spawn(C[i], [C[i-1]] if i > 0 else [])
            def subtask():
                sleep(0.05) # Required delay to allow out of order execution without dependencies
                task_results.append(i)

    sleep_until(lambda: len(task_results) == 10)
    assert task_results == list(range(10))


def test_placement(runtime_sched):
    devices = [cpu(0), cpu(1), cpu(6)]
    for rep in repetitions():
        task_results = []
        for (i, dev) in enumerate(devices):
            @spawn(placement=dev)
            def task():
                task_results.append(get_current_devices()[0])
            sleep_until(lambda: len(task_results) == i+1)

        assert task_results == devices


def test_placement_data(runtime_sched):
    try:
        from parla.cuda import gpu
    except:
        skip("Test needs cuda.")
        return
    devices = [cpu(0), gpu(0)]
    for rep in repetitions():
        task_results = []
        for (i, dev) in enumerate(devices):
            d = dev.memory()(np.array([1, 2, 3]))
            @spawn(placement=d)
            def task():
                task_results.append(get_current_devices()[0])
            sleep_until(lambda: len(task_results) == i+1)

        assert task_results == devices


def test_placement_options_vcus(runtime_sched):
    # test multiple options in placement list with only one device used in the end
    for rep in repetitions():
        N = 4
        task_results = []
        for i in range(N):
            @spawn(placement=[cpu(0), cpu(1)], vcus=1)
            def task():
                sleep(0.1)
                task_results.append(get_current_devices()[0])
        sleep_until(lambda: len(task_results) == N)
        assert set(task_results) == {cpu(0), cpu(1)}
        assert task_results.count(cpu(0)) == N/2
        assert task_results.count(cpu(1)) == N/2


def test_placement_options_memory(runtime_sched):
    # test multiple options in placement list with only one device used in the end
    for rep in repetitions():
        task_results = []
        for i in range(4):
            @spawn(placement=[cpu(0), cpu(3)], memory=cpu(0).available_memory)
            def task():
                sleep(0.1)
                task_results.append(get_current_devices()[0])
        sleep_until(lambda: len(task_results) == 4)
        assert set(task_results) == {cpu(0), cpu(3)}
        assert task_results.count(cpu(0)) == 2
        assert task_results.count(cpu(3)) == 2


def test_placement_multi():
    # Dummy environments with no components for testing.
    environments = [TaskEnvironment(placement=d, components=[]) for d in combinations(cpu.devices, 2)]
    with Parla(environments):
        devices = [frozenset((cpu(0), cpu(1))), frozenset((cpu(1), cpu(2))), frozenset((cpu(6), cpu(3)))]
        for rep in repetitions():
            task_results = []
            for (i, dev) in enumerate(devices):
                @spawn(placement=dev, ndevices=2)
                def task():
                    task_results.append(frozenset(get_current_devices()))
                sleep_until(lambda: len(task_results) == i+1)

            assert task_results == devices


def test_placement_await(runtime_sched):
    devices = [cpu(0), cpu(1), cpu(6)]

    for rep in repetitions():
        task_results = []
        for (i, dev) in enumerate(devices):
            @spawn(placement=dev)
            async def task():
                task_results.append(get_current_devices()[0])
                await tasks() # Await nothing to force a new task.
                task_results.append(get_current_devices()[0])
            sleep_until(lambda: len(task_results) == (i+1)*2)

        assert task_results == [cpu(0), cpu(0), cpu(1), cpu(1), cpu(6), cpu(6)]


def test_memory_aware_scheduling(runtime_sched):
    # test memory restrictions
    for rep in repetitions():
        task_results = []
        for i in range(8):
            @spawn(placement=cpu, memory=cpu(0).available_memory)
            def task():
                task_results.append(get_current_devices()[0])
                sleep(0.1)
        sleep_until(lambda: len(task_results) == 8)
        assert 8 >= len(set(task_results)) >= 4


def test_architecture(runtime_sched):
    task_results = []
    @spawn(placement=cpu)
    def task():
        task_results.append(1)
    @spawn(placement=cpu)
    def task():
        task_results.append(1)
    @spawn(placement=cpu)
    def task():
        task_results.append(1)

    sleep_until(lambda: len(task_results) == 3)
    assert task_results == [1]*3


def test_architecture_multiple(runtime_sched):
    try:
        from parla.cuda import gpu
    except:
        skip("Test needs cuda.")
        return
    try:
        task_results = set()
        @spawn(placement=cpu)
        async def task():
            task_results.add(get_current_devices()[0].architecture.id)
        @spawn(placement=gpu)
        async def task():
            task_results.add(get_current_devices()[0].architecture.id)

        sleep_until(lambda: len(task_results) == 2)
        assert task_results == {"cpu", "gpu"}
    except ValueError as e:
        skip(str(e))


