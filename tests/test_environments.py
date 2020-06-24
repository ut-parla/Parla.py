import logging
import threading
from time import sleep

from parla import Parla, TaskEnvironment
from parla.cpu import cpu
from parla.environments import EnvironmentComponentInstance, EnvironmentComponentDescriptor
from parla.tasks import *

logger = logging.getLogger(__name__)

def repetitions():
    """Return an iterable of the repetitions to perform for probabilistic/racy tests."""
    return range(5)


def sleep_until(predicate, timeout=2, period=0.05):
    """Sleep until either `predicate()` is true or 2 seconds have passed."""
    for _ in range(int(timeout/period)):
        if predicate():
            break
        sleep(period)
    assert predicate(), "sleep_until timed out ({}s)".format(timeout)


thread_locals = threading.local()

class DummyComponentInstance(EnvironmentComponentInstance):
    def __init__(self, descriptor: "DummyComponent", env: TaskEnvironment):
        super().__init__(descriptor)
        self.value = descriptor.value

    def __enter__(self):
        assert not hasattr(thread_locals, "value")
        thread_locals.value = self.value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert thread_locals.value == self.value
        del thread_locals.value
        return False

class DummyComponent(EnvironmentComponentDescriptor):
    def __init__(self, value):
        super(DummyComponent, self).__init__()
        self.value = value

    def combine(self, other):
        assert isinstance(other, DummyComponent)
        return DummyComponent(self.value + other.value)

    def __call__(self, env: TaskEnvironment) -> DummyComponentInstance:
        return DummyComponentInstance(self, env)


def test_dummy_environment_component():
    # Dummy environments with no components for testing.
    environments = [TaskEnvironment(placement=[cpu(0)], components=[DummyComponent("test")])]
    with Parla(environments):
        task_results = []
        @spawn()
        def task():
            assert get_current_devices() == [cpu(0)]
            task_results.append(thread_locals.value)

        sleep_until(lambda: len(task_results) == 1)
        assert task_results == ["test"]


def test_multiple_environments_fixed_assignment():
    # Dummy environments with no components for testing.
    environments = [TaskEnvironment(placement=[cpu(0)], components=[DummyComponent("foo")]),
                    TaskEnvironment(placement=[cpu(1)], components=[DummyComponent("bar")])]
    with Parla(environments):
        task_results = []
        @spawn(placement=cpu(0))
        def task():
            task_results.append(thread_locals.value)
        @spawn(placement=cpu(1))
        def task():
            task_results.append(thread_locals.value)

        sleep_until(lambda: len(task_results) == 2)
        assert set(task_results) == {"foo", "bar"}


def test_multiple_environments_free_assignment():
    # Dummy environments with no components for testing.
    environments = [TaskEnvironment(placement=[cpu(0)], components=[DummyComponent("foo")]),
                    TaskEnvironment(placement=[cpu(1)], components=[DummyComponent("bar")])]
    with Parla(environments):
        for _ in repetitions():
            task_results = []
            @spawn(vcus=1)
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)
            @spawn(vcus=1)
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)
            @spawn(vcus=1)
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)

            sleep_until(lambda: len(task_results) == 3)
            assert set(task_results) == {"foo", "bar"}


def test_multiple_environments_tagged():
    # Dummy environments with no components for testing.
    environments = [TaskEnvironment(placement=[cpu(0)], components=[DummyComponent("foo")], tags=(threading,)),
                    TaskEnvironment(placement=[cpu(1)], components=[DummyComponent("bar")], tags=(logging,))]
    with Parla(environments):
        for _ in repetitions():
            task_results = []
            @spawn(tags=(threading,))
            def task():
                task_results.append(thread_locals.value)

            sleep_until(lambda: len(task_results) == 1)
            assert task_results == ["foo"]

            task_results = []
            @spawn(tags=(logging,))
            def task():
                task_results.append(thread_locals.value)

            sleep_until(lambda: len(task_results) == 1)
            assert task_results == ["bar"]


def test_multiple_environments_best_fit():
    # Dummy environments with no components for testing.
    environments = [TaskEnvironment(placement=[cpu(0)], components=[DummyComponent("foo")]),
                    TaskEnvironment(placement=[cpu(0), cpu(1)], components=[DummyComponent("bar")])]
    with Parla(environments):
        for _ in repetitions():
            task_results = []
            @spawn(placement=cpu(0))
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)
            @spawn(placement=cpu(0))
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)
            @spawn(placement=cpu(0))
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)

            sleep_until(lambda: len(task_results) == 3)
            assert task_results == ["foo", "foo", "foo"]

            task_results = []
            @spawn(placement=cpu(1))
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)
            @spawn(placement=cpu(1))
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)
            @spawn(placement=cpu(1))
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)

            sleep_until(lambda: len(task_results) == 3)
            assert task_results == ["bar", "bar", "bar"]


def test_multiple_environments_less_good_fit():
    # Dummy environments with no components for testing.
    environments = [TaskEnvironment(placement=[cpu(0), cpu(1)], components=[DummyComponent("foo")]),
                    TaskEnvironment(placement=[cpu(2), cpu(3), cpu(4)], components=[DummyComponent("bar")])]
    with Parla(environments):
        for _ in repetitions():
            task_results = []
            # The first two will fit in the the first environment using 0.5 of the environment.
            # The next two will spill into the less good (0.33) fit of the second environment.
            @spawn(placement=[cpu(1), cpu(2)], vcus=1)
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)
            @spawn(placement=[cpu(1), cpu(2)], vcus=1)
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)
            @spawn(placement=[cpu(1), cpu(2)], vcus=1)
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)
            @spawn(placement=[cpu(1), cpu(2)], vcus=1)
            def task():
                sleep(0.1)
                task_results.append(thread_locals.value)

            sleep_until(lambda: len(task_results) == 4)
            task_results.sort()
            assert task_results == ["bar", "bar", "foo", "foo"]
