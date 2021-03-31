import abc
import logging
import random
from abc import abstractmethod, ABCMeta
from collections import deque, namedtuple, defaultdict
from contextlib import contextmanager
import threading
import time
from itertools import combinations
from numbers import Number
from threading import Thread, Condition
from typing import Optional, Collection, Union, Dict, List, Any, Tuple, FrozenSet, Iterable, TypeVar

from .device import get_all_devices, Device, Architecture
from .environments import TaskEnvironmentRegistry, TaskEnvironment

logger = logging.getLogger(__name__)

__all__ = ["Task", "SchedulerContext", "DeviceSetRequirements", "OptionsRequirements", "ResourceRequirements"]

# TODO: This module is pretty massively over-engineered the actual use case could use a much simpler scheduler.

_ASSIGNMENT_FAILURE_WARNING_LIMIT = 32


# Note: tasks can be implemented as lock free, however, atomics aren't really a thing in Python, so instead
# make each task have its own lock to mimic atomic-like counters for dependency tracking.


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


ResourceDict = Dict[str, Union[float, int]]


class ResourceRequirements(object, metaclass=abc.ABCMeta):
    __slots__ = ["resources", "ndevices", "tags"]

    tags: FrozenSet[Any]
    resources: ResourceDict
    ndevices: int

    def __init__(self, resources: ResourceDict, ndevices: int, tags: Collection[Any]):
        assert all(isinstance(v, str) for v in resources.keys())
        assert all(isinstance(v, (float, int)) for v in resources.values())
        self.resources = resources
        self.ndevices = ndevices
        self.tags = frozenset(tags)

    @property
    def possibilities(self) -> Iterable["ResourceRequirements"]:
        return [self]

    @property
    def exact(self):
        return False

    @abstractmethod
    def __parla_placement__(self):
        raise NotImplementedError()


class EnvironmentRequirements(ResourceRequirements):
    __slots__ = ["environment"]
    environment: TaskEnvironment

    def __init__(self, resources: ResourceDict, environment: TaskEnvironment, tags: Collection[Any]):
        super().__init__(resources, len(environment.placement), tags)
        self.environment = environment

    @property
    def devices(self):
        return self.environment.placement

    @property
    def exact(self):
        return True

    def __parla_placement__(self):
        return self.environment.__parla_placement__()

    def __repr__(self):
        return "EnvironmentRequirements({}, {})".format(self.resources, self.environment)


class DeviceSetRequirements(ResourceRequirements):
    __slots__ = ["devices"]
    devices: FrozenSet[Device]

    def __init__(self, resources: ResourceDict, ndevices: int, devices: Collection[Device], tags: Collection[Any]):
        super().__init__(resources, ndevices, tags)
        assert devices
        assert all(isinstance(dd, Device) for dd in devices)
        self.devices = frozenset(devices)
        assert len(self.devices) >= self.ndevices

    @property
    def possibilities(self) -> Iterable["DeviceSetRequirements"]:
        return (DeviceSetRequirements(self.resources, self.ndevices, ds, self.tags)
                for ds in combinations(self.devices, self.ndevices))

    @property
    def exact(self):
        return len(self.devices) == self.ndevices

    def __parla_placement__(self):
        return self.devices

    def __repr__(self):
        return "DeviceSetRequirements({}, {}, {}, exact={})".format(self.resources, self.ndevices, self.devices, self.exact)


class OptionsRequirements(ResourceRequirements):
    __slots__ = ["options"]
    options: List[List[Device]]

    def __init__(self, resources, ndevices, options, tags: Collection[Any]):
        super().__init__(resources, ndevices, tags)
        assert len(options) > 1
        assert all(isinstance(a, Device) for a in options)
        self.options = options

    @property
    def possibilities(self) -> Iterable[DeviceSetRequirements]:
        return (opt
                for ds in self.options
                for opt in DeviceSetRequirements(self.resources, self.ndevices, ds, self.tags).possibilities)

    def __parla_placement__(self):
        return list(set(d for ds in self.options for d in ds))

    def __repr__(self):
        return "OptionsRequirements({}, {}, {})".format(self.resources, self.ndevices, self.options)


class Task:
    assigned: bool
    req: ResourceRequirements
    _dependees: List["Task"]
    _state: TaskState

    def __init__(self, func, args, dependencies: Collection["Task"], taskid,
                 req: ResourceRequirements, name: Optional[str] = None):
        self.name = name
        self._mutex = threading.Lock()
        with self._mutex:
            self.taskid = taskid

            self.req = req
            self.assigned = False

            self._state = TaskRunning(func, args, None)
            self._dependees = []

            get_scheduler_context().incr_active_tasks()

            self._set_dependencies(dependencies)

            # Expose the self reference to other threads as late as possible, but not after potentially getting
            # scheduled.
            taskid.task = self
            
            logger.debug("Task %r: Creating", self)

            self._check_remaining_dependencies()

    @property
    def dependees(self) -> Tuple["Task"]:
        """
        A tuple of the currently known tasks that depend on self.

        This tuple may be added to at any time during the life of a task (as dependee tasks are created),
        but tasks are never removed.
        """
        return tuple(self._dependees)

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

    def _complete_dependency(self):
        with self._mutex:
            self._remaining_dependencies -= 1
            self._check_remaining_dependencies()

    def _check_remaining_dependencies(self):
        if not self._remaining_dependencies:
            logger.info("Task %r: Scheduling", self)
            get_scheduler_context().enqueue_task(self)

    def _add_dependee(self, dependee: "Task"):
        """Add the dependee if self is not completed, otherwise return False."""
        with self._mutex:
            if self._state.is_terminal:
                return False
            else:
                self._dependees.append(dependee)
                return True

    def run(self):
        ctx = get_scheduler_context()
        task_state = TaskException(RuntimeError("Unknown fatal error"))
        assert self.assigned, "Task was not assigned before running."
        assert isinstance(self.req, EnvironmentRequirements), \
            "Task was not assigned a specific environment requirement before running."
        try:
            # Allocate the resources used by this task (blocking)
            for d in self.req.devices:
                ctx.scheduler._available_resources.allocate_resources(d, self.req.resources, blocking=True)
            # Run the task and assign the new task state
            try:
                assert isinstance(self._state, TaskRunning)
                # We both set the environment as a thread local using _environment_scope, and enter the environment itself.
                with _scheduler_locals._environment_scope(self.req.environment), self.req.environment:
                    task_state = self._state.func(self, *self._state.args)
                if task_state is None:
                    task_state = TaskCompleted(None)
            except Exception as e:
                task_state = TaskException(e)
                logger.exception("Exception in task")
            finally:
                logger.info("Finally for task %r", self)
                # Deallocate all the resources, both from the allocation above and from the "assignment" done by
                # the scheduler.
                for d in self.req.devices:
                    ctx.scheduler._available_resources.deallocate_resources(d, self.req.resources)
                    ctx.scheduler._unassigned_resources.deallocate_resources(d, self.req.resources)
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
        return "<Task {} nrem_deps={} state={} req={req} assigned={assigned}>".format(self.name or "", self._remaining_dependencies, type(self._state).__name__, **self.__dict__)

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
    def spawn_task(self, function, args, deps, taskid, req, name: Optional[str] = None):
        return Task(function, args, deps, taskid, req, name)

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
    _environment: Optional[TaskEnvironment]
    _scheduler_context_stack: List[SchedulerContext]

    def __init__(self):
        super(_SchedulerLocals, self).__init__()
        self._scheduler_context_stack = []
        self._environment = None

    @property
    def environment(self):
        if self._environment:
            return self._environment
        else:
            raise InvalidSchedulerAccessException("TaskEnvironment not set in this context")

    @contextmanager
    def _environment_scope(self, env: TaskEnvironment):
        self._environment = env
        try:
            yield
        finally:
            self._environment = None

    @property
    def scheduler_context(self) -> SchedulerContext:
        if self._scheduler_context_stack:
            return self._scheduler_context_stack[-1]
        else:
            raise InvalidSchedulerAccessException("No scheduler is available in this context")


_scheduler_locals = _SchedulerLocals()


def get_scheduler_context() -> SchedulerContext:
    return _scheduler_locals.scheduler_context


def get_devices() -> Collection[Device]:
    return _scheduler_locals.environment.placement


class ControllableThread(Thread, metaclass=ABCMeta):
    _should_run: bool
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
        self._status = "Initializing"

    @property
    def scheduler(self) -> "Scheduler":
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
                        logger.debug("Getting a task: %r", self)
                        return self._queue.pop()
                    else:
                        return None
                except IndexError:
                    logger.debug("Blocking for a task: %r (%s)", self, self._monitor)
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
                for component in self.scheduler.components:
                    component.initialize_thread()
                while self._should_run:
                    self._status = "Getting Task"
                    task: Task = self._pop_task()
                    if not task:
                        break
                    self._status = "Running Task {}".format(task)
                    task.run()
        except Exception as e:
            logger.exception("Unexpected exception in Task handling")
            self.scheduler.stop()

    def dump_status(self, lg=logger):
        lg.info("%r:\n%r", self, self._queue)

    def __repr__(self):
        return "<{} {} {}>".format(type(self).__name__, self.index, self._status)


class ResourcePool:
    _multiplier: float
    _monitor: Condition
    _devices: Dict[Device, Dict[str, float]]

    # Resource pools track device resources. Environments are a separate issue and are not tracked here. Instead,
    # tasks will consume resources based on their devices even though those devices are bundled into an environment.

    def __init__(self, multiplier):
        self._multiplier = multiplier
        self._monitor = threading.Condition(threading.Lock())
        self._devices = self._initial_resources(multiplier)

    @staticmethod
    def _initial_resources(multiplier):
        return {dev: {name: amt * multiplier for name, amt in dev.resources.items()} for dev in get_all_devices()}

    def allocate_resources(self, d: Device, resources: ResourceDict, *, blocking: bool = False) -> bool:
        """Allocate the resources described by `dd`.

        :param d: The device on which resources exist.
        :param resources: The resources to allocate.
        :param blocking: If True, this call will block until the resource is available and will always return True.

        :return: True iff the allocation was successful.
        """
        return self._atomically_update_resources(d, resources, -1, blocking)

    def deallocate_resources(self, d: Device, resources: ResourceDict) -> None:
        """Deallocate the resources described by `dd`.

        :param d: The device on which resources exist.
        :param resources: The resources to deallocate.
        """
        ret = self._atomically_update_resources(d, resources, 1, False)
        assert ret

    def _atomically_update_resources(self, d: Device, resources: ResourceDict, multiplier, block: bool):
        with self._monitor:
            to_release = []
            success = True
            for name, v in resources.items():
                if not self._update_resource(d, name, v * multiplier, block):
                    success = False
                    break
                else:
                    to_release.append((name, v))
            else:
                to_release.clear()

            logger.info("Attempted to allocate %s * %r (blocking %s) => %s\n%r", multiplier, (d, resources), block, success, self)
            if to_release:
                logger.info("Releasing resources due to failure: %r", to_release)

            for name, v in to_release:
                ret = self._update_resource(d, name, -v * multiplier, block)
                assert ret

            assert not success or len(to_release) == 0 # success implies to_release empty
            return success

    def _update_resource(self, dev: Device, res: str, amount: float, block: bool):
        try:
            while True: # contains return
                dres = self._devices[dev]
                if -amount <= dres[res]:
                    dres[res] += amount
                    if amount > 0:
                        self._monitor.notify_all()
                    assert dres[res] <= dev.resources[res] * self._multiplier, \
                        "{}.{} was over deallocated".format(dev, res)
                    assert dres[res] >= 0, \
                        "{}.{} was over allocated".format(dev, res)
                    return True
                else:
                    if block:
                        self._monitor.wait()
                    else:
                        return False
        except KeyError:
            raise ValueError("Resource {}.{} does not exist".format(dev, res))

    def __repr__(self):
        return "ResourcePool(devices={})".format(self._devices)


class AssignmentFailed(Exception):
    pass

_T = TypeVar('_T')
def shuffled(lst: Iterable[_T]) -> List[_T]:
    """Shuffle a list non-destructively."""
    lst = list(lst)
    random.shuffle(lst)
    return lst

class Scheduler(ControllableThread, SchedulerContext):
    _environments: TaskEnvironmentRegistry
    _worker_threads: List[WorkerThread]
    _unassigned_resources: ResourcePool
    _available_resources: ResourcePool
    period: float
    max_worker_queue_depth: int

    def __init__(self, environments: Collection[TaskEnvironment], n_threads: int = None, period: float = 0.01,
                 max_worker_queue_depth: int = 2):
        super().__init__()
        n_threads = n_threads or len(environments)
        self._environments = TaskEnvironmentRegistry(*environments)
        self._exceptions = []
        self._active_task_count = 1 # Start with one count that is removed when the scheduler is "exited"
        self.max_worker_queue_depth = max_worker_queue_depth
        self.period = period
        self._monitor = threading.Condition(threading.Lock())
        self._allocation_queue = deque()
        self._available_resources = ResourcePool(multiplier=1.0)
        self._unassigned_resources = ResourcePool(multiplier=max_worker_queue_depth*1.0)
        self._worker_threads = [WorkerThread(self, i) for i in range(n_threads)]
        for t in self._worker_threads:
            t.start()
        self.start()

    @property
    def components(self) -> List["EnvironmentComponentInstance"]:
        return [i for e in self._environments for i in e.components.values()]

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
        for t in self._worker_threads:
            t.join()
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

    def _try_assignment(self, req: EnvironmentRequirements) -> bool:
        # Allocate available resources
        allocated_devices: List[Device] = []
        try:
            for d in shuffled(req.devices):
                assert len(allocated_devices) < req.ndevices
                assert isinstance(d, Device)
                if self._unassigned_resources.allocate_resources(d, req.resources):
                    allocated_devices.append(d)
                else:
                    raise AssignmentFailed()
            # Select an environment the matches the allocated resources.
            return True
        except AssignmentFailed:
            # Free any resources we already assigned
            for d in allocated_devices:
                self._unassigned_resources.deallocate_resources(d, req.resources)
            return False

    def _assignment_policy(self, task: Task):
        """
        Attempt to assign resources to `task`.

        If this function returns true, `task.req` should have type EnvironmentRequirements.

        :return: True if the assignment succeeded, False otherwise.
        """
        # Build a list of environments with "qualities" assigned based on how well they match a possible
        # option for the task
        env_match_quality = defaultdict(lambda: 0)
        for opt in shuffled(task.req.possibilities):
            if isinstance(opt, DeviceSetRequirements):
                for e in self._environments.find_all(placement=opt.devices, tags=opt.tags, exact=False):
                    intersection = e.placement & opt.devices
                    match_quality = len(intersection) / len(e.placement)
                    env_match_quality[e] = max(env_match_quality[e], match_quality)
            elif isinstance(opt, EnvironmentRequirements):
                env_match_quality[opt.environment] = max(env_match_quality[opt.environment], 1)
        environments_to_try = list(env_match_quality.keys())
        environments_to_try.sort(key=env_match_quality.__getitem__, reverse=True)
        # print(task, ":", env_match_quality, "  ", environments_to_try)

        # Try the environments in order
        specific_requirements = None
        for env in environments_to_try:
            specific_requirements = EnvironmentRequirements(task.req.resources, env, task.req.tags)
            if self._try_assignment(specific_requirements):
                task.req = specific_requirements
                return True

        return False

    def run(self) -> None:
        # noinspection PyBroadException
        try: # Catch all exception to report them usefully
            while self._should_run:
                task: Optional[Task] = self._dequeue_task()
                if not task:
                    # Exit if the dequeue fails. This implies a failure or shutdown.
                    break

                if not task.assigned:
                    is_assigned = self._assignment_policy(task)
                    assert isinstance(is_assigned, bool)
                    task.assigned = is_assigned

                # assert task.req.exact == task.assigned
                assert not task.assigned or isinstance(task.req, EnvironmentRequirements)

                if not task.assigned:
                    task._assignment_tries = getattr(task, "_assignment_tries", 0) + 1
                    if task._assignment_tries > _ASSIGNMENT_FAILURE_WARNING_LIMIT:
                        logger.warning("Task %r: Failed to assign devices. The required resources may not be "
                                       "available on this machine at all.\n"
                                       "Available resources: %r\n"
                                       "Unallocated resources: %r",
                                       task, self._available_resources, self._unassigned_resources)
                    # Put task we cannot assign resources to at the back of the queue
                    logger.debug("Task %r: Failed to assign", task)
                    self.enqueue_task(task)
                    # Avoid spinning when no tasks are schedulable.
                    time.sleep(self.period)
                    # TODO: There is almost certainly a better way to handle this. Add a dependency on
                    #  a task holding the needed resources?
                else:
                    # Place task in shortest worker queue if it's not too long
                    while True:  # contains break
                        worker = min(self._worker_threads, key=lambda w: w.estimated_queue_depth())
                        if worker.estimated_queue_depth() < self.max_worker_queue_depth:
                            logger.debug("Task %r: Enqueued on worker %r", task, worker)
                            worker._enqueue_task_local(task)
                            break
                        else:
                            # Delay a bit waiting for a workers queue to shorten; This is not an issue since
                            # definitionally there is plenty of work in the queues.
                            time.sleep(self.period)
        except Exception:
            logger.exception("Unexpected exception in Scheduler")
            self.stop()

    def stop(self):
        super().stop()
        for w in self._worker_threads:
            w.stop()

    def report_exception(self, e: BaseException):
        with self._monitor:
            self._exceptions.append(e)

    def dump_status(self, lg=logger):
        lg.info("%r:\n%r\nunassigned: %r\navailable: %r", self,
                self._allocation_queue, self._unassigned_resources, self._available_resources)
        w: WorkerThread
        for w in self._worker_threads:
            w.dump_status(lg)
