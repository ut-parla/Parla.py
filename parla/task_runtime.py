import abc
import logging
import random
from abc import abstractmethod, ABCMeta
from collections import deque, namedtuple
from contextlib import contextmanager
import threading
import time
from threading import Thread
from typing import Optional, Collection

from .device import get_all_devices, Device, Architecture

logger = logging.getLogger(__name__)

__all__ = []


# Note: tasks can be implemented as lock free, however,
# atomics aren't really a thing in Python, so instead
# make each task have its own lock to mimic atomic-like
# counters for dependency tracking.


TaskAwaitTasks = namedtuple("AwaitTasks", ("dependencies", "value_task"))


class TaskState(object, metaclass=abc.ABCMeta):
    __slots__ = []

    @property
    @abstractmethod
    def is_terminal(self):
        raise NotImplementedError()


class TaskRunning(TaskState):
    __slots__ = ["func", "args", "dependencies"]

    @property
    def is_terminal(self):
        return False

    def __init__(self, func, args, dependencies):
        if dependencies is not None and (
                not isinstance(dependencies, Collection) or not all(isinstance(d, Task) for d in dependencies)):
            raise ValueError("dependencies must be a collection of Tasks")
        self.dependencies = dependencies
        self.args = args
        self.func = func

    def clear_dependencies(self):
        self.dependencies = None

    def __repr__(self):
        return "TaskRunning({}, {}, {})".format(self.func.__name__, self.args, self.dependencies)


class TaskCompleted(TaskState):
    __slots__ = ["ret"]

    def __init__(self, ret):
        self.ret = ret

    @property
    def is_terminal(self):
        return True

    def __repr__(self):
        return "TaskCompleted({})".format(self.ret)


class TaskException(TaskState):
    __slots__ = ["exc"]

    @property
    def is_terminal(self):
        return True

    def __init__(self, exc):
        self.exc = exc

    def __repr__(self):
        return "TaskException({})".format(self.exc)


class Task:
    _state: TaskState

    def __init__(self, func, args, dependencies: Collection["Task"], taskid,
                 resources, reads, writes, cost, constraints):
        self._mutex = threading.Lock()
        with self._mutex:
            self.taskid = taskid

            self._state = TaskRunning(func, args, None)

            self.constraints = constraints
            self.cost = cost
            self.writes = writes
            self.reads = reads
            self.resources = resources
            assert self.resources
            assert all(self.resources)

            self._dependees = []
            # self._result = None
            # self._exception = None
            self._assigned_device = None
            self.assigned_amount = 0

            get_scheduler_context().incr_active_tasks()

            self._set_dependencies(dependencies)

            # Expose the self reference to other threads as late as possible, but not after potentially getting
            # scheduled.
            taskid.task = self

            self._check_remaining_dependencies()

            # logger.info("Task %r: Creating", self)

    def _set_dependencies(self, dependencies):
        self._remaining_dependencies = len(dependencies)
        for dep in dependencies:
            if not dep._add_dependee(self):
                self._remaining_dependencies -= 1

    @property
    def result(self):
        if isinstance(self._state, TaskCompleted):
            return self._state.ret
        elif isinstance(self._state, TaskException):
            raise self._state.exc

    @property
    def assigned_device(self):
        return self._assigned_device

    @assigned_device.setter
    def assigned_device(self, dev):
        if self._assigned_device is not None and self._assigned_device != dev:
            raise ValueError("The device cannot be assigned more than once.")
        self._assigned_device = dev

    def _complete_dependency(self):
        with self._mutex:
            self._remaining_dependencies -= 1
            self._check_remaining_dependencies()

    def _check_remaining_dependencies(self):
        if not self._remaining_dependencies:
            logger.info("Task %r: Scheduling", self)
            get_scheduler_context().enqueue_task(self)

    def _add_dependee(self, dependee):
        """Add the dependee if self is not completed, otherwise return False."""
        with self._mutex:
            if self._state.is_terminal:
                return False
            else:
                self._dependees.append(dependee)
                return True

    def run(self):
        if not self.assigned_device:
            raise ValueError("Task was not assigned before running. This requirement will be fixed in the future.")
        ctx = get_scheduler_context()
        task_state = TaskException(RuntimeError("Unknown fatal error"))
        ctx.scheduler._available_resources.allocate_resources_blocking(self.assigned_device, self.assigned_amount)
        try:
            with _scheduler_locals._device_scope(self.assigned_device):
                try:
                    assert isinstance(self._state, TaskRunning)
                    task_state = self._state.func(self, *self._state.args)
                    if task_state is None:
                        task_state = TaskCompleted(None)
                except Exception as e:
                    task_state = TaskException(e)
                finally:
                    ctx.scheduler._available_resources.allocate_resources(self.assigned_device, -self.assigned_amount)
                    ctx.scheduler._unassigned_resources.allocate_resources(self.assigned_device, -self.assigned_amount)
                    self._set_state(task_state)
        except Exception as e:
            logger.exception("Task %r: Exception in task handling", self)
            raise e

    def _notify_dependees(self):
        with self._mutex:
            for dependee in self._dependees:
                dependee._complete_dependency()

    def __await__(self):
        return (yield TaskAwaitTasks([self], self))

    def __repr__(self):
        return "<Task nrem_deps={_remaining_dependencies} state={_state}>".format(**self.__dict__)

    def _set_state(self, new_state: TaskState):
        # old_state = self._state
        logger.info("Task %r: %r -> %r", self, self._state, new_state)
        self._state = new_state
        ctx = get_scheduler_context()

        if isinstance(new_state, TaskException):
            ctx.scheduler.report_exception(new_state.exc)
        elif isinstance(new_state, TaskRunning):
            self._set_dependencies(new_state.dependencies)
            self._check_remaining_dependencies()
            new_state.clear_dependencies()

        if new_state.is_terminal:
            self._notify_dependees()
            ctx.decr_active_tasks()


class InvalidSchedulerAccessException(RuntimeError):
    pass


class SchedulerContext(metaclass=ABCMeta):
    def spawn_task(self, function, args, deps, taskid, resources, reads, writes, cost, constraints):
        return Task(function, args, deps, taskid,
            resources=resources, reads=reads, writes=writes,
            cost=cost, constraints=constraints)

    @abstractmethod
    def enqueue_task(self, task):
        raise NotImplementedError()

    def __enter__(self):
        _scheduler_locals._scheduler_context_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _scheduler_locals._scheduler_context_stack.pop()

    @property
    @abstractmethod
    def scheduler(self) -> "Scheduler":
        raise NotImplementedError()

    @abstractmethod
    def incr_active_tasks(self):
        raise NotImplementedError()

    @abstractmethod
    def decr_active_tasks(self):
        raise NotImplementedError()


class _SchedulerLocals(threading.local):
    def __init__(self):
        super(_SchedulerLocals, self).__init__()
        self._scheduler_context_stack = []

    @property
    def device(self):
        if hasattr(self, "_device"):
            return self._device
        else:
            raise InvalidSchedulerAccessException("Device not set in this context")

    @contextmanager
    def _device_scope(self, device):
        self._device = device
        try:
            yield
        finally:
            self._device = None

    @property
    def scheduler_context(self) -> SchedulerContext:
        if self._scheduler_context_stack:
            return self._scheduler_context_stack[-1]
        else:
            raise InvalidSchedulerAccessException("No scheduler is available in this context")


_scheduler_locals = _SchedulerLocals()


def get_scheduler_context() -> SchedulerContext:
    return _scheduler_locals.scheduler_context


def get_device() -> Device:
    return _scheduler_locals.device


class ControllableThread(Thread, metaclass=ABCMeta):
    _monitor: threading.Condition

    def __init__(self):
        super().__init__()
        self._should_run = True

    def stop(self):
        with self._monitor:
            self._should_run = False
            self._monitor.notify_all()

    @abstractmethod
    def run(self):
        pass


class WorkerThread(ControllableThread, SchedulerContext):
    def __init__(self, scheduler, index):
        super().__init__()
        self._monitor = threading.Condition(threading.Lock())
        self.index = index
        self._scheduler = scheduler
        # Use a deque to store local tasks (a high-performance implementation would a work stealing optimized deque).
        # In this implementation the right is the "local-end", so append/pop are used by this worker and
        # appendleft/popleft are used by the scheduler or other workers.
        self._queue = deque()
        self.start()

    @property
    def scheduler(self):
        return self._scheduler

    def incr_active_tasks(self):
        self.scheduler.incr_active_tasks()

    def decr_active_tasks(self):
        self.scheduler.decr_active_tasks()

    def estimated_queue_depth(self):
        """Return the current estimated depth of this workers local queue.

        This should be considered immediately stale and may in fact be slightly wrong (+/- 1 element) w.r.t. to any
        real value of the queue depth. These limitations are to allow high-performance queue implementations that
        don't provide an atomic length operation.
        """
        # Assume this will not FAIL due to concurrent access, slightly incorrect results are not an issue.
        return len(self._queue)

    def _pop_task(self):
        """Pop a task from the queue head.
        """
        with self._monitor:
            while True:
                try:
                    if self._should_run:
                        return self._queue.pop()
                    else:
                        return None
                except IndexError:
                    self._monitor.wait()

    def steal_task_nonblocking(self):
        """Pop a task from the queue tail.

        :return: The task for None if no task was available to steal.
        """
        with self._monitor:
            try:
                return self._queue.popleft()
            except IndexError:
                return None

    def _push_task(self, task):
        """Push a local task on the queue head.
        """
        with self._monitor:
            self._queue.append(task)
            self._monitor.notify()

    def enqueue_task(self, task):
        """Push a task on the queue tail.
        """
        # For the moment, bypass the local queue and put the task in the global scheduler queue
        self.scheduler.enqueue_task(task)
        # Allowing local resource of tasks (probably only when it comes to the front of the queue) would allow threads
        # to make progress even if the global scheduler is blocked by other assignment tasks. However, it would also
        # require that the workers do some degree of resource assignment which complicates things and could break
        # correctness or efficiency guarantees. That said a local, "fast assignment" algorithm to supplement the
        # out-of-band assignment of the scheduler would probably allow Parla to efficiently run programs with
        # significantly finer-grained tasks.

        # For tasks that are already assigned it may be as simple as:
        #     self.scheduler._unassigned_resources.allocate_resources(task.assigned_device, task.assigned_amount)
        #     self._push_task(task)
        # This would need to fail over to the scheduler level enqueue if the resources is not available for assignment.

    def _enqueue_task_local(self, task):
        with self._monitor:
            self._queue.appendleft(task)
            self._monitor.notify()

    def run(self) -> None:
        try:
            with self:
                while self._should_run:
                    task: Task = self._pop_task()
                    if not task:
                        break
                    task.run()
        except Exception as e:
            logger.exception("Unexpected exception in Task handling")
            self.scheduler.stop()


class AvailableResources:
    def __init__(self, multiplier=1):
        self._multiplier = multiplier
        self._monitor = threading.Condition(threading.Lock())
        self._available = {res: res.amount_available*multiplier for res in get_all_devices()}

    def allocate_resources(self, specific_resource, amount) -> bool:
        """Allocate (or deallocate with a negative amount) a resource.

        :return: True iff the allocation was successful.
        """
        with self._monitor:
            try:
                if amount <= self._available[specific_resource]:
                    self._available[specific_resource] -= amount
                    if amount < 0:
                        self._monitor.notify_all()
                    assert self._available[specific_resource] <= specific_resource.amount_available * self._multiplier,\
                        "{} was over deallocated".format(specific_resource)
                    assert self._available[specific_resource] >= 0, "{} was over allocated".format(specific_resource)
                    return True
                else:
                    return False
            except KeyError:
                raise ValueError("Resource {} does not exist".format(specific_resource))

    def allocate_resources_blocking(self, specific_resource, amount) -> None:
        """Allocate (or deallocate with a negative amount) a resource.
        """
        with self._monitor:
            try:
                self._monitor.wait_for(lambda: amount <= self._available[specific_resource])
                self._available[specific_resource] -= amount
                assert self._available[specific_resource] <= specific_resource.amount_available * self._multiplier, \
                    "{} was over deallocated".format(specific_resource)
                assert self._available[specific_resource] >= 0, "{} was over allocated".format(specific_resource)
            except KeyError:
                raise ValueError("Resource {} does not exist".format(specific_resource))

    def available(self):
        return dict(self._available)


class Scheduler(ControllableThread, SchedulerContext):
    def __init__(self, n_threads, period=0.01, max_worker_queue_depth=2):
        super().__init__()
        self._exceptions = []
        self._active_task_count = 1 # Start with one count that is removed when the scheduler is "exited"
        self.max_worker_queue_depth = max_worker_queue_depth
        self.period = period
        self._monitor = threading.Condition(threading.Lock())
        self._allocation_queue = deque()
        self._available_resources = AvailableResources()
        self._unassigned_resources = AvailableResources(multiplier=max_worker_queue_depth)
        logger.info("Initializing scheduler with multiplier %d and resources %r",
                     max_worker_queue_depth, self._available_resources.available())
        self._worker_threads = [WorkerThread(self, i) for i in range(n_threads)]
        self._should_run = True
        self.start()

    @property
    def scheduler(self):
        return self

    def __enter__(self):
        if self._active_task_count != 1:
            raise InvalidSchedulerAccessException("Schedulers can only have a single scope.")
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.decr_active_tasks()
        with self._monitor:
            while self._should_run:
                self._monitor.wait()
        if self._exceptions:
            # TODO: Should combine all of them into a single exception.
            raise self._exceptions[0]

    def incr_active_tasks(self):
        with self._monitor:
            self._active_task_count += 1

    def decr_active_tasks(self):
        done = False
        with self._monitor:
            self._active_task_count -= 1
            if self._active_task_count == 0:
                done = True
        if done:
            self.stop()

    def enqueue_task(self, task: Task):
        """Enqueue a task on the resource allocation queue.
        """
        with self._monitor:
            self._allocation_queue.appendleft(task)
            self._monitor.notify_all()

    def _dequeue_task(self, timeout=None) -> Optional[Task]:
        """Dequeue a task from the resource allocation queue.
        """
        with self._monitor:
            while True:
                try:
                    if self._should_run:
                        return self._allocation_queue.pop()
                    else:
                        return None
                except IndexError:
                    self._monitor.wait(timeout)
                    if timeout is not None:
                        try:
                            return self._allocation_queue.pop()
                        except IndexError:
                            return None

    def run(self) -> None:
        try:
            while self._should_run:
                task = self._dequeue_task()
                if not task:
                    break

                # Select resource for task
                device_to_use = None
                amount_to_use = 0
                logger.debug("Task %r: Beginning assignment with %r", task,self._unassigned_resources.available())
                if task.assigned_device:
                    # If the task is already assigned then just allocate the resource if possible and move on.
                    if self._unassigned_resources.allocate_resources(task.assigned_device, task.assigned_amount):
                        logger.debug("Task %r: Allocated assigned resource %r %d", task,
                                    task.assigned_device, task.assigned_amount)
                        device_to_use = task.assigned_device
                        amount_to_use = task.assigned_amount
                else:
                    for res_set in task.resources:
                        for (res, amt) in res_set.items():
                            res: Architecture
                            devices = res.devices
                            random.shuffle(devices)  # Shuffle the devices to avoid over assigning the early devices
                            for dev in devices:
                                if self._unassigned_resources.allocate_resources(dev, amt):
                                    if task.constraints(dev):
                                        logger.info("Task %r: Allocated resource %r %d", task, dev, amt)
                                        device_to_use = dev
                                        amount_to_use = amt
                                        break
                                    else:
                                        # Return the resource to the pool if it is rejected.
                                        self._unassigned_resources.allocate_resources(dev, -amt)
                                        logger.debug("Task %r: Task rejected resource %r %d (returned to pool)", task, dev, amt)
                            # TODO: Currently we only support a single device and a single resource

                if not device_to_use:
                    # Put task we cannot assign resources to at the back of the queue
                    self.enqueue_task(task)
                    # Avoid spinning when no tasks are schedulable.
                    time.sleep(self.period)
                else:
                    # Assign resource
                    task.assigned_device = device_to_use
                    task.assigned_amount = amount_to_use
                    # Place task in shortest worker queue if it's not too long
                    while True:  # contains break
                        worker = min(self._worker_threads, key=lambda w: w.estimated_queue_depth())
                        if worker.estimated_queue_depth() < self.max_worker_queue_depth:
                            logger.debug("Task %r: Enqueued on worker %r", task, worker)
                            worker._enqueue_task_local(task)
                            break
                        else:
                            # Delay a bit waiting for a workers queue to shorten
                            time.sleep(self.period)
        except Exception as e:
            logger.exception("Unexpected exception in Scheduler")
            self.stop()

    def stop(self):
        super().stop()
        for w in self._worker_threads:
            w.stop()

    def report_exception(self, e: BaseException):
        with self._monitor:
            self._exceptions.append(e)
