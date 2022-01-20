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

#logging.basicConfig(level = logging.INFO)
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
    def env_no(self):
        return self.environment.env_no

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


class TaskID:
    pass


class TaskBase:
    pass


class Task(TaskBase):
    # This flag specifies if a task is assigned device.
    # If it is, it sets to True. Otherwise, it sets to False.
    # Any thread could read this flag, and therefore, mutex
    # is always required.
    assigned: bool

    def __init__(self, func, args, dependencies: Collection["Task"], taskid,
                 req: ResourceRequirements,
                 # TODO(lhc): task will get input/output data information
                 # through decorator parameters. These would not string types,
                 # but data wrapper types.
                 # For now, assume that those information is passed here.
                 #in_data_list: List[str],
                 #out_data_list: List[str],
                 name: Optional[str] = None):
        self._mutex = threading.Lock()
        with self._mutex:
            self._name = name
            self._taskid = taskid
            self._req = req
            self.assigned = False

            self._state = TaskRunning(func, args, None)
            self._dependees = []
            # Maintain dependenceis as a list object.
            # Therefore, bi-directional edges exist among
            # dependent tasks.
            # These dependencies are moved to a data movement
            # task.
            self._set_dependencies(dependencies)
            #self.in_data_list = in_data_list
            #self.out_data_list = out_data_list
            # Expose the self reference to other threads as late as possible, but not after potentially getting
            # scheduled.
            taskid.task = self
            
            logger.debug("Task %r: Creating", self)
            get_scheduler_context().incr_active_tasks()
            # Enqueue this task right after spawning on the spawend queue.
            # The task could have dependencies.
            get_scheduler_context().enqueue_spawned_task(self)

    @property
    def result(self):
        if isinstance(self._state, TaskCompleted):
            return self._state.ret
        elif isinstance(self._state, TaskException):
            raise self._state.exc

    def _reset_dependencies(self):
        self._dependencies = []

    @property
    def taskid(self) -> TaskID:
        return self._taskid

    @property
    def name(self) -> str:
        return self._name

    @property
    def req(self):
        return self._req

    @req.setter
    def req(self, new_req):
        self._req = new_req

    @property
    def dependencies(self) -> Tuple["TaskBase"]:
        return tuple(self._dependencies)

    @property
    def dependees(self) -> Tuple["TaskBase"]:
        """
        A tuple of the currently known tasks that depend on self.

        This tuple may be added to at any time during the life of a task (as dependee tasks are created),
        but tasks are never removed.
        """
        return tuple(self._dependees)

    def set_assigned(self):
        with self._mutex:
            self.assigned = True

    def is_assigned(self):
        with self._mutex:
            if self.assigned:
                return True
            else:
                return False

    def _set_dependencies(self, dependencies):
        self._dependencies = dependencies
        self._remaining_dependencies = len(dependencies)
        for dep in dependencies:
            if not dep._add_dependee(self):
                self._remaining_dependencies -= 1

    def _complete_dependency(self):
        with self._mutex:
            self._remaining_dependencies -= 1
            self._check_remaining_dependencies()
            print("Computation task, ", self.name, "'s one dependency is completed. [remaining:", \
                  self._remaining_dependencies, "]", sep='')

    def _check_remaining_dependencies(self):
        if not self._remaining_dependencies and self.assigned:
            logger.info("Task %r: Scheduling", self)
            get_scheduler_context().enqueue_task(self)

    def bool_check_remaining_dependencies(self):
        if not self._remaining_dependencies:
            return False
        else:
            return True

    def _add_dependee(self, dependee: "TaskBase"):
        """Add the dependee if self is not completed, otherwise return False."""
        with self._mutex:
            if self._state.is_terminal:
                return False
            else:
                print("Computation task, ", self.name, ", added a dependee, ", dependee, sep='')
                self._dependees.append(dependee)
                return True

    def _notify_dependees(self):
        with self._mutex:
            for dependee in self._dependees:
                dependee._complete_dependency()

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
                self._set_state(task_state)
        except Exception as e:
            logger.exception("Task %r: Exception in task handling", self)
            raise e

    def __await__(self):
        return (yield TaskAwaitTasks([self], self))

    def __repr__(self):
        return "<Task {} nrem_deps={} state={} req={_req} assigned={assigned}>".format(self.name or "", self._remaining_dependencies, type(self._state).__name__, **self.__dict__)


class DataMovementTask(TaskBase):
    # This flag specifies if a task is assigned device.
    # If it is, it sets to True. Otherwise, it sets to False.
    # Any thread could read this flag, and therefore, mutex
    # is always required.
    assigned: bool

    # TODO(lhc): For now, input and output data are string.
    #            For now, this class performs no-op.
    def __init__(self, dependencies: Collection["TaskBase"],
            computation_task: Task, taskid,
             req: ResourceRequirements,
            in_data_list: List[str],
            out_data_list: List[str], name: Optional[str] = None):
        self._mutex = threading.Lock()
        with self._mutex:
            self._name = name
            self._taskid = taskid
            self._req = req
            # This task is an auxiliary task
            # which is created at mapping task subgraph
            # construction. This task is always assigned the target task.
            self.assigned = True
            # The source (computation) task becomes a dependee of
            # this data movement task.
            # The dependees are set by `_set_dependencies()`.
            self._dependees = []
            # Data movement task gets dependencies of the
            # source (computation) task.
            # TODO(lhc): Both a data movement task and a kernel
            #            task are dependees of the dependent tasks.
            #            Therefore, when the dependent tasks are completed,
            #            they iterate both type tasks, which is inefficient.
            #            This is because I don't want to touch data race
            #            until this inefficient version works stable.
            self._set_dependencies(dependencies)

            # Input and output data should be passed to move data
            # if necessary.
            # TODO(lhc): For now, it is unclear how data is managed.
            #            For now, use a string list tmporarily.
            self.in_data_list = in_data_list
            self.out_data_list = out_data_list
            # TODO(lhc): temporary task running state.
            #            This would be a data movement kernel.
            self._state = TaskRunning(None, None, None)

    @property
    def taskid(self) -> TaskID:
        return self._taskid

    @property
    def name(self) -> str:
        return self._name

    @property
    def req(self):
        return self._req

    @req.setter
    def req(self, new_req):
        self._req = new_req

    @property
    def dependencies(self) -> Tuple["TaskBase"]:
        return tuple(self._dependencies)

    @property
    def dependees(self) -> Tuple["TaskBase"]:
        """
        A tuple of the currently known tasks that depend on self.

        This tuple may be added to at any time during the life of a task (as dependee tasks are created),
        but tasks are never removed.
        """
        return tuple(self._dependees)

    def set_assigned(self):
        with self._mutex:
            self.assigned = True

    def is_assigned(self):
        with self._mutex:
            if self.assigned:
                return True
            else:
                return False

    def _complete_dependency(self):
        with self._mutex:
            self._remaining_dependencies -= 1
            self._check_remaining_dependencies()
            print("Data-movement task, ", self.name, "'s one dependency is " \
                  "completed. [remaining:", self._remaining_dependencies, sep='')

    def _check_remaining_dependencies(self):
        if not self._remaining_dependencies and self.assigned:
            logger.info("Task %r: Scheduling", self)
            get_scheduler_context().enqueue_task(self)

    def bool_check_remaining_dependencies(self):
        if not self._remaining_dependencies:
            return False
        else:
            return True

    def _notify_dependees(self):
        with self._mutex:
            for dependee in self._dependees:
                dependee._complete_dependency()

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

    def run(self):
        ctx = get_scheduler_context()
        task_state = None
        # XXX(lhc)
        #task_state = TaskException(RuntimeError("Unknown fatal error"))
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
                    # TODO(lhc): func() is not yet implemented.
                    #            we should create data movement kernel manually.

                    #task_state = self._state.func(self, *self._state.args)
                    print("Data movement start: move from ", self.in_data_list, \
                            " to ", self.out_data_list, sep='')

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
                self._set_state(task_state)
        except Exception as e:
            logger.exception("Task %r: Exception in task handling", self)
            raise e

    def _add_dependee(self, dependee: "TaskBase"):
        """Add the dependee if self is not completed, otherwise return False."""
        with self._mutex:
            if self._state.is_terminal:
                return False
            else:
                print("Data-movement task, ", self.name, ", added a dependee, ", dependee, sep='')
                self._dependees.append(dependee)
                return True

    def _set_dependencies(self, dependencies):
        self._remaining_dependencies = len(dependencies)
        for dep in dependencies:
            if not dep._add_dependee(self):
                self._remaining_dependencies -= 1

    def __await__(self):
        return (yield TaskAwaitTasks([self], self))

    def __repr__(self):
        return "<Task {} nrem_deps={} state={} assigned={assigned}>".format(self.name or "", self._remaining_dependencies, type(self._state).__name__, **self.__dict__)


class _TaskLocals(threading.local):
    def __init__(self):
        super(_TaskLocals, self).__init__()
        self.task_scopes = []

    @property
    def ctx(self):
        return getattr(self, "_ctx", None)

    @ctx.setter
    def ctx(self, v):
        self._ctx = v

    @property
    def global_tasks(self):
        return getattr(self, "_global_tasks", [])

    @global_tasks.setter
    def global_tasks(self, v):
        self._global_tasks = v


task_locals = _TaskLocals()


class TaskID:
    """The identity of a task.

    This combines some ID value with the task object itself. The task
    object is assigned by `spawn`. This can be used in place of the
    task object in most places.

    """
    _task: Optional[Task]
    _id: Iterable[int]

    def __init__(self, name, id: Iterable[int]):
        """"""
        self._name = name
        self._id = id
        self._task = None

    @property
    def task(self):
        """Get the `Task` associated with this ID.

        :raises ValueError: if there is no such task.
        """
        if not self._task:
            raise ValueError("This task has not yet been spawned so it cannot be used.")
        return self._task

    @task.setter
    def task(self, v):
        assert not self._task
        self._task = v

    @property
    def id(self):
        """Get the ID object.
        """
        return self._id

    @property
    def name(self):
        """Get the space name.
        """
        return self._name

    @property
    def full_name(self):
        """Get the space name.
        """
        return "_".join(str(i) for i in (self._name, *self._id))

    def __repr__(self):
        return "TaskID({}, task={})".format(self.full_name, self._task)

    def __str__(self):
        return "<TaskID {}>".format(self.full_name)

    def __await__(self):
        return (yield TaskAwaitTasks([self.task], self.task))


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
                    print("[WorkerThread ", self.index,"] Blocking for a task.", sep='')
                    self._monitor.wait()
                    print("[WorkerThread ", self.index,"] Waking up.", sep='')

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

    def enqueue_spawned_task(self, task: Task):
        self.scheduler.enqueue_spawned_task(task)

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
                    print("Worker thread start:", task.name, sep='')
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
    # Worker threads for each device.
    _worker_threads: List[WorkerThread]
    _available_resources: ResourcePool
    period: float
    max_worker_queue_depth: int

    def __init__(self, environments: Collection[TaskEnvironment], n_threads: int = None, period: float = 0.01,
                 max_worker_queue_depth: int = 2):
        super().__init__()
        # TODO(lhc): for now, assume that n_threads is always None.
        #            Each device needs a dedicated thread.
        n_threads = sum(d.resources.get("vcus", 1) for e in environments for d in e.placement)
        self._environments = TaskEnvironmentRegistry(*environments)
        self._exceptions = []
        self._active_task_count = 1 # Start with one count that is removed when the scheduler is "exited"
        self.max_worker_queue_depth = max_worker_queue_depth
        self.period = period
        self._monitor = threading.Condition(threading.Lock())
        self._allocation_queue = deque()
        # Spawned queue should consist of two levels, current and new.
        # New spawned tasks or tasks failed to schedule are always
        # enqueued on the new queue.
        # Tasks which the scheduler will try to schedule at the current
        # iteration are always dequeued from the current queue.
        # For each iteration, elements on the new queue are moved and
        # appended to the current queue.
        # Through the two level queues, the scheduler could avoid tailing elements,
        # and move to the next super step for the mapped tasks.
        self._spawned_task_queue = deque()
        self._new_spawned_task_queue = deque()
        self._mapped_task_queue = deque()
        self._new_mapped_task_queue = deque()
        self._device_queues = []
        self._device_queues = [deque() for n in range(0, n_threads)]
        self._available_resources = ResourcePool(multiplier=1.0)
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

    def enqueue_spawned_task(self, task: Task):
        """Enqueue a spawned task on the spawned task queue.
           Scheduler iterates the queue and assigns resources
           regardless of remaining dependencies.
        """
        with self._monitor:
            self._new_spawned_task_queue.appendleft(task)

    def _dequeue_spawned_task(self) -> Optional[Task]:
        """Dequeue a task from the spawned task queue.
        """
        with self._monitor:
            # Try to dequeue a task and if there is no
            # remaining task, returns None.
            # TODO(lhc): if a task is spawned later
            #            than this function,
            #            when that will be popped?
            #            scheduler is busy waiting this queue?
            try:
                return self._spawned_task_queue.pop()
            except IndexError:
                return None

    def enqueue_mapped_task(self, task: Task):
        """Enqueue a task on the mapped subgraph queue.
           Note that this enqueueing has no data race.
        """
        self._new_mapped_task_queue.appendleft(task)

    def _dequeue_mapped_task(self) -> Optional[Task]:
        try:
            return self._mapped_task_queue.pop()
        except IndexError:
            return None

    def enqueue_task(self, task: Task):
        """Enqueue a task on the resource allocation queue.
           Note that this enqueue has no data race.
        """
        self._allocation_queue.appendleft(task)

    def _dequeue_task(self, timeout=None) -> Optional[Task]:
        """Dequeue a task from the resource allocation queue.
        """
        while True:
            try:
                if self._should_run:
                    return self._allocation_queue.pop()
                else:
                    return None
            except IndexError:
                # Keep proceeding the next step.
                return None

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

        # Try the environments in order
        # Environment is registered device environments.
        # This mainly specifies device types.
        # resources is memory resources of the corresponding devices.
        # This loop only checks if devices meet task constraints or not.
        for env in environments_to_try:
            is_res_constraint_satisifed = True
            for d in shuffled(env.placement):
                for name, amount in task.req.resources.items():
                    if d.resources[name] < amount:
                        is_res_constraint_satisifed = False
                        break
                if not is_res_constraint_satisifed:
                    break
            if is_res_constraint_satisifed:
                task.req = EnvironmentRequirements(task.req.resources, env, task.req.tags)
                return True
        return False

    def fill_curr_spawned_task_queue(self):
        """ It moves tasks on the new spawned task queue to
            the current queue.
        """
        with self._monitor:
            new_q = self._new_spawned_task_queue
            new_tasks = [new_q.popleft() for _ in range(len(new_q))]
            if len(new_tasks) > 0:
                self._spawned_task_queue.extendleft(new_tasks)

    def fill_curr_mapped_task_queue(self):
        """ It moves tasks on the new mapped task queue to
            the current queue.
        """
        with self._monitor:
            new_q = self._new_mapped_task_queue
            new_tasks = [new_q.popleft() for _ in range(len(new_q))]
            if len(new_tasks) > 0:
                self._mapped_task_queue.extendleft(new_tasks)

    def run_mapper(self):
        # The first loop iterates a spawned task queue
        # and constructs a mapped task subgrpah.
        while True:
            self.fill_curr_spawned_task_queue()
            task: Optional[Task] = self._dequeue_spawned_task()
            if task:
                if not task.assigned:
                    print("\n[Scheduler] Spawned task, ", task.name, ", is mapped.", sep='')
                    is_assigned = self._assignment_policy(task)
                    assert isinstance(is_assigned, bool)
                    if not is_assigned:
                        self.enqueue_spawned_task(task)
                    else:
                        # Create data movement task
                        # This step consists of five steps.
                        #
                        # 1. Get dependencies of the task.
                        # 2. Reset dependency lists of the task.
                        #    (Therefore, we can reduce unnecessary
                        #     dependee iterations when higher priority tasks
                        #     are done)
                        # 3. Create data movement task with the depdencies.
                        # 4. Set the original task as the data task's dependee.
                        # 5. Set the data task as the original task's denpdency.
                        deps = task.dependencies
                        # TODO(lhc): I am not confident if this task id and
                        #            local/global tasks work correctly.
                        taskid = TaskID("global_" + str(len(task_locals.global_tasks)),
                                        (len(task_locals.global_tasks),))
                        task_locals.global_tasks += [taskid]
                        taskid.dependencies = deps
                        task._reset_dependencies()
                        datamove_task = DataMovementTask(deps, task,
                                             taskid, task.req,
                                             ["in1 processing", "in2 processing"],
                                             ["out processing"],
                                             task.name + ".datamovement"
                                             );
                        self.incr_active_tasks()
                        task._set_dependencies([datamove_task])
                        # Only computation needs to set a assigned flag.
                        # Data movement task is set as assigned when it is created.
                        task.set_assigned()
                        # If a task has no dependency after it is assigned to devices,
                        # immediately enqueue a corresponding data movement task to
                        # the ready queue.
                        if not datamove_task.bool_check_remaining_dependencies():
                            self.enqueue_task(datamove_task)
                else:
                    logger.exception("[Scheduler] Tasks on the spawned Q ", \
                                     "should be not assigned any device.")
                    self.stop()
            else:
                # If there is no spawned task at this moment,
                # move to the mapped task scheduling.
                break

    def run_deviceq_mapper(self):
        while True:
            task: Optional[TaskBase] = self._dequeue_task()
            if not task or not task.assigned:
                logger.debug("Task %r: Failed to assign", task)
                break
            for d in task.req.devices:
                print("[Scheduler] Task, ", task.name, ", is enqueued onto a device Q.", sep='')
                worker = self._worker_threads[task.req.env_no]
                worker._enqueue_task_local(task)

    def run(self) -> None:
        # noinspection PyBroadException
        try: # Catch all exception to report them usefully
            while self._should_run:
                self.run_mapper()
                self.run_deviceq_mapper()
        except Exception:
            logger.exception("Unexpected exception in Scheduler")
            self.stop()

    def stop(self):
        super().stop()
        for w in self._worker_threads:
            w.stop()

    def report_exception(self, e: BaseException):
        with self._monitor:
            print("Report exception:", e)
            self._exceptions.append(e)

    def dump_status(self, lg=logger):
        lg.info("%r:\n%r\navailable: %r", self,
                self._allocation_queue, self._available_resources)
        w: WorkerThread
        for w in self._worker_threads:
            w.dump_status(lg)
