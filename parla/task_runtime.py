# General imports
from functools import lru_cache
import logging
import random
from abc import abstractmethod, ABCMeta
from collections import deque, namedtuple, defaultdict
from contextlib import contextmanager
from enum import Enum
import threading
import time
from itertools import combinations
from typing import Optional, Collection, Union, Dict, List, Any, Tuple, FrozenSet, Iterable, TypeVar, Deque

# Parla imports
from parla.device import get_all_devices, Device
from parla.environments import TaskEnvironmentRegistry, TaskEnvironment
from parla.cpu_impl import cpu

# Logger configuration (uncomment and adjust level if needed)
#logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["Task", "SchedulerContext", "DeviceSetRequirements", "OptionsRequirements", "ResourceRequirements", "get_current_devices"]

# Note: tasks can be implemented as lock free, however, atomics aren't really a thing in Python, so instead
# make each task have its own lock to mimic atomic-like counters for dependency tracking.

TaskAwaitTasks = namedtuple("AwaitTasks", ("dependencies", "value_task"))


class UnspawnedDependencies:
    """ Collection of dependencies where the upstream tasks ("predecessors") are not spawned yet
    and thus making downstream tasks ("successors") wait.
    """

    # A dictionary with predecessors as keys and (lists of) successors as values,
    # i.e. value tasks are waiting for key tasks.
    #
    # Example:
    #   Predecessor A -> [Successor B, Successor C]
    #
    # - When A is spawned, it should notify B and C.
    # - B and C must have already been spawned.
    _dependencies: Dict['TaskID', List['TaskID']]

    def __init__(self):
        self._mutex = threading.Lock()
        self._dependencies = defaultdict(list)

    def add(self, predecessor: 'TaskID', successor: 'TaskID'):
        with self._mutex:
            self._dependencies[predecessor].append(successor)

    def get_successors(self, tid: 'TaskID') -> List['TaskID']:
        with self._mutex:
            return self._dependencies[tid]


unspawned_dependencies = UnspawnedDependencies()


class TaskState(object, metaclass=ABCMeta):
    __slots__ = []

    @property
    @abstractmethod
    def is_terminal(self):
        raise NotImplementedError()


class TaskWaiting(TaskState):
    """ This state specifies that a task is waiting for dependencies' spawnings
    """
    @property
    def is_terminal(self):
        return False


#TODO(lhc): Why do we need dependency information at here?
#           It is not exploited/managed correctly.
class TaskRunning(TaskState):
    __slots__ = ["func", "args", "dependencies"]

    @property
    def is_terminal(self):
        return False

    # The argument dependencies intentially has no hint.
    # But its corresponding member instance value is declared as list.
    # Callers can pass None if they want to pass empty dependencies.
    def __init__(self, func, args, dependencies: Optional[List]):
        if dependencies is not None:
            # d could be one of four types: Task, DataMovementTask,
            # TaskID or other types.
            # Task and DataMovementTask are expected types and
            # are OK to be in the dependency list.
            # TaskID is not yet spawned, and will be added as a
            # Task when it is spawned.
            # (Please refer to tasks.py:_task_callback() for detiailed
            #  information)
            #
            # Other types are not allowed and not expected.
            assert all(isinstance(d, (Task, TaskID)) for d in dependencies)
            self.dependencies = [d for d in dependencies if isinstance(d, Task)]
        else:
            self.dependencies = []
        self.args = args
        self.func = func

    def clear_dependencies(self):
        self.dependencies = None

    def __repr__(self):
        if self.func:
            #return "TaskRunning({}, {}, {})".format(self.func.__name__, self.args, self.dependencies)
            return "TaskRunning({})".format(self.func.__name__)
        else:
            return "Functionless task"


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


class ResourceRequirements(object, metaclass=ABCMeta):
    """
    When a task spawns, it has a set of requirements based on parameters
    supplied to @spawn.
    This class represents those resources.
    This is an Abstract Base Class - see below for classes which inherit from it.
    Currently, spawned tasks only use DeviceSetRequirements.
    After mapping, tasks receive EnvironmentRequirements.
    As of writing this comment, idk what the difference is. Enviroments seem unnecessary and confusing.
    OptionsRequirements aren't even used anywhere at all.
    """
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


# This basically stores all the devices a task is *permitted* to run on,
# taking into account spawn's placement parameter
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


# CURRENTLY NOT USED
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
        return (
            opt for ds in self.options for opt in DeviceSetRequirements(
                self.resources, self.ndevices, ds, self.tags
            ).possibilities
        )

    def __parla_placement__(self):
        return list(set(d for ds in self.options for d in ds))

    def __repr__(self):
        return "OptionsRequirements({}, {}, {})".format(self.resources, self.ndevices, self.options)


class Task:
    def __init__(self, dependencies: Collection["Task"], taskid,
                 req: ResourceRequirements, name: Optional[str] = None):
        self._mutex = threading.Lock()
        with self._mutex:
            # This is the name of the task, which is distinct from the TaskID. It inherits its name from the func.
            self._name = name
            self._successors = []
            self._taskid = taskid
            # Maintain dependencies as a list object.
            # Therefore, bi-directional edges exist among dependent tasks.
            # Some of these dependencies are moved to a data movement task.
            self._set_dependencies(dependencies)
            self._req = req
            # This flag specifies if a task is assigned device.
            # If it is, it sets to True. Otherwise, it sets to False.
            self.assigned = False
            # Dependent tasks' event objects for runahead scheduling.
            self.dependent_events = []
            self.events = []

    @property
    def taskid(self) -> 'TaskID':
        return self._taskid

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def req(self):
        return self._req

    @req.setter
    def req(self, new_req):
        self._req = new_req

    @property
    def dependencies(self) -> Tuple["Task"]:
        with self._mutex:
            return self._predecessors

    @property
    def dependees(self) -> Tuple["Task"]:
        """
        A tuple of the currently known tasks that depend on self.

        This tuple may be added to at any time during the life of a task
        (as dependee tasks are created), but tasks are never removed.
        """
        return tuple(self._successors)

    @property
    def result(self):
        if isinstance(self._state, TaskCompleted):
            return self._state.ret
        elif isinstance(self._state, TaskException):
            raise self._state.exc

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
        self._predecessors = dependencies
        self._remaining_dependencies = len(dependencies)
        for dep in dependencies:
            # If a dependency is TaskID, not Task object,
            # it implies that it is not yet spawned.
            # Ignore it.
            if isinstance(dep, TaskID):
                continue
            if not dep._add_dependee(self):
                self._remaining_dependencies -= 1

    def _set_dependencies_mutex(self, dependencies):
        with self._mutex:
            return self._set_dependencies(dependencies)

    def _check_remaining_dependencies(self):
        if not self._remaining_dependencies and self.assigned:
            logger.info("[Task] %s: Scheduling", str(self.taskid))
            get_scheduler_context().enqueue_task(self)

    def _check_remaining_dependencies_mutex(self):
        with self._mutex:
            if not self._remaining_dependencies and self.assigned:
                logger.info("Task %r: Scheduling", self)
                get_scheduler_context().enqueue_task(self)

    def bool_check_remaining_dependencies(self):
        if not self._remaining_dependencies:
            return False
        else:
            return True

    def bool_check_remaining_dependencies_mutex(self):
        with self._mutex:
            if not self._remaining_dependencies:
                return False
            else:
                return True

    def is_dependent(self, cand: "Task"):
        with self._mutex:
            if cand in self._predecessors:
                return True
            else:
                return False

    def _add_dependee_mutex(self, dependee: "Task"):
        """Add the dependee if self is not completed, otherwise return False."""
        with self._mutex:
            if self._state.is_terminal:
                return False
            else:
                logger.debug("[Task] %s added a dependee, %s",
                             self.name, dependee.name)
                self._successors.append(dependee)
                return True

    def _add_dependee(self, dependee: "Task"):
        """Add the dependee if self is not completed, otherwise return False."""
        if self._state.is_terminal:
            return False
        else:
            logger.debug("Task, %s added a dependee, %s",
                         self.name, dependee)
            self._successors.append(dependee)
            return True

    def _notify_dependees_mutex(self, events=None):
        with self._mutex:
            for dependee in self._successors:
                dependee._complete_dependency(events)
            self._successors = []

    def _notify_dependees(self, events=None):
        for dependee in self._successors:
            dependee._complete_dependency(events)
        self._successors = []

    def _add_dependency_mutex(self, dependency):
        with self._mutex:
            return self._add_dependency(dependency)

    def _add_dependency(self, dependency):
        self._remaining_dependencies += 1
        self._predecessors.append(dependency)
        if not dependency._add_dependee_mutex(self):
            self._remaining_dependencies -= 1
            return False
        return True

    def _complete_dependency(self, events):
        with self._mutex:
            self._remaining_dependencies -= 1
            # Add events from one dependent task.
            # (We are aiming to multiple device tasks, and it would
            #  be possible to have multiple events)
            if (events is not None):
                self.dependent_events.append(events)
            self._check_remaining_dependencies()
            logger.info(f"[Task %s] Task dependency completed. \
                (remaining: %d)", self.name, self._remaining_dependencies)

    def _set_state(self, new_state: TaskState):
        # old_state = self._state
        logger.info("[Task] %r: %r -> %r", str(self._taskid), self._state, new_state)
        self._state = new_state
        ctx = get_scheduler_context()

        if isinstance(new_state, TaskException):
            ctx.scheduler.report_exception(new_state.exc)
        elif isinstance(new_state, TaskRunning):
            self._set_dependencies(new_state.dependencies)
            self._check_remaining_dependencies()
            new_state.clear_dependencies()
        if new_state.is_terminal:
            ctx.decr_active_tasks()

    def __await__(self):
        return (yield TaskAwaitTasks([self], self))

    def __repr__(self):
        return "<Task {} nrem_deps={} state={} assigned={assigned}>". \
               format(self.name or "", self._remaining_dependencies,
                      type(self._state).__name__, **self.__dict__)


class ComputeTask(Task):
    def __init__(self, func, args, dependencies: Collection["Task"], taskid: 'TaskID',
                 req: ResourceRequirements, dataflow: "Dataflow",
                 name: Optional[str] = None,
                 num_unspawned_deps: int = 0):
        super(ComputeTask, self).__init__(dependencies, taskid, req, name)
        with self._mutex:
            # This task could be spawend when it is ready.
            # To set its state Running when it is running later,
            # store functions and arguments as member variables.
            self._func = func
            self._args = args
            self.dataflow = dataflow  # input/output/inout of the task
            # Expose the self reference to other threads as late as possible,
            # but not after potentially getting scheduled.
            taskid.task = self

            self.num_unspawned_predecessors = num_unspawned_deps
            # If this task is not waiting for any dependent tasks,
            # enqueue onto the spawned queue.
            if self.num_unspawned_predecessors == 0:
                self._activate()
            else:
                self._state = TaskWaiting()
            logger.debug("Task %r: Creating", self)

    def __notify_spawned_successors(self):
        """ Notify all successors who wait for this task.
        PRIVATE USE ONLY. Not thread-safe and should be called WITH ITS MUTEX.
        """
        # Get all the waiting dependee list from the global collection.
        successors = unspawned_dependencies.get_successors(self.taskid)
        for d_tid in successors:
            dt = d_tid.task
            if dt is None:
                raise ValueError("The dependee task is None:", str(d_tid))
            assert isinstance(dt, ComputeTask)
            dt.decr_num_unspawned_deps(self)
            self._successors.append(dt)

    def _activate(self):
        assert self.num_unspawned_predecessors == 0
        self.__notify_spawned_successors()
        self._state = TaskRunning(self._func, self._args, None)
        get_scheduler_context().incr_active_tasks() # TODO(bozhi): integrate with enqueue
        # Enqueue this task right after spawning on the spawend queue.
        # The task could have dependencies.
        get_scheduler_context().enqueue_spawned_task(self)

    def decr_num_unspawned_deps(self, predecessor: "Task"):
        with self._mutex:
            self.num_unspawned_predecessors -= 1
            self._remaining_dependencies += 1
            self._predecessors.append(predecessor)
            if self.num_unspawned_predecessors == 0:
                self._activate()

    def run(self):
        ctx = get_scheduler_context()
        task_state = TaskException(RuntimeError("Unknown fatal error"))
        assert self.assigned, "Task was not assigned before running."
        assert isinstance(self.req, EnvironmentRequirements), \
            "Task was not assigned a specific environment requirement before running."
        try:
            # Run the task and assign the new task state
            try:
                assert isinstance(self._state, TaskRunning)
                # TODO(lhc): This assumes Parla only has two devices.
                #            The reason why I am trying to do is importing
                #            Parla's cuda.py is expensive.
                #            Whenever we import cuda.py, cupy compilation
                #            is invoked. We should remove or avoid that.

                # First, create device/stream/event instances.
                # Second, gets the created event instance.
                # Third, it scatters the event to dependees who wait for
                # the current task.
                env = self.req.environment
                with _scheduler_locals._environment_scope(env), env:
                    self.events = env.get_events_from_components()
                    if len(self.dependent_events) > 0:
                        # Only wait events if any remaining dependent tasks exist.
                        # TODO(lhc): This does not support nested CPU tasks from GPU tasks.
                        #            When we notify dependees, we don't know places on
                        #            which the dependees will run since those would decide
                        #            at the next step, mapping step.
                        #            Therefore, the current runtime does not consider
                        #            dependees' placements, but just creates events
                        #            on the devices where the current task is running.
                        #            For example, when CPU tasks get GPU events,
                        #            they will just skip and will not wait for them.
                        #            This is not technical limitation, just need refactoring
                        #            and adding more features.
                        #            But our target applications do not have this pattern
                        #            and will do it later.
                        env.wait_dependent_events(self.dependent_events)
                    # Already waited all dependent events.
                    # Initialize the dependent event list.
                    self.dependent_events = []
                    task_state = self._state.func(self, *self._state.args)
                    env.record_events()
                    if len(self.events) > 0:
                        # If any event created by the current task exist,
                        # notify dependees and make them wait for that event,
                        # not Parla task completion.
                        self._notify_dependees_mutex(self.events)
                    env.sync_events()
                if task_state is None:
                    task_state = TaskCompleted(None)
            except Exception as e:
                task_state = TaskException(e)
                logger.exception("Exception in task")
            finally:
                logger.info("Finally for task %r", self)

                # Deallocate resources
                for d in self.req.devices:
                    for resource, amount in self.req.resources.items():
                        logger.debug("Task %r deallocating %d %s from device %r", self, amount, resource, d)
                    ctx.scheduler._available_resources.deallocate_resources(d, self.req.resources)
                    ctx.scheduler._device_compute_task_counts[d] -= 1

                # Update parray tracking information
                # The easiest way to do this is just untrack and re-track the array for anything that could've changed
                for parray in (self.dataflow.inout + self.dataflow.output):
                    # TODO (ses): Make this atomic so the GIL can't switch between
                    ctx.scheduler._available_resources.untrack_parray(parray)
                    ctx.scheduler._available_resources.track_parray(parray)

                # Protect the case that it notifies dependees and
                # any dependee task is spawned before setting state.
                with self._mutex:
                    # Regardless of the previous notification,
                    # (So, before leaving the current run(), the above)
                    # it should notify dependees since
                    # new dependees could be added after the above
                    # notifications, while other devices are running
                    # their kernels asynchronously.
                    self._notify_dependees()
                    self._set_state(task_state)

        except Exception as e:
            logger.exception("Task %r: Exception in task handling", self)
            raise e


class OperandType(Enum):
    IN = 0
    OUT = 1
    INOUT = 2


class DataMovementTask(Task):
    def __init__(self, computation_task: ComputeTask, taskid,
                 req: ResourceRequirements, target_data,
                 operand_type: OperandType, name: Optional[str] = None):
        super(DataMovementTask, self).__init__([], taskid, req, name)
        with self._mutex:
            # A data movement task is created after mapping phase.
            # Therefore, this class is already assigned to devices.
            self.assigned = True
            self._target_data = target_data
            self._operand_type = operand_type
            # TODO(lhc): temporary task running state.
            #            This would be a data movement kernel.
            self._state = TaskRunning(None, None, None)

    def run(self):
        logger.debug(f"[DataMovementTask %s] Starting", self.name)
        ctx = get_scheduler_context()
        # TODO(lhc)
        #task_state = TaskException(RuntimeError("Unknown fatal error"))
        assert self.assigned, "Task was not assigned before running."
        assert isinstance(self.req, EnvironmentRequirements), \
            "Task was not assigned a specific environment requirement before running."

        try:
            # Run the task and assign the new task state
            try:
                assert isinstance(self._state, TaskRunning)
                # First, create device/stream/event instances.
                # Second, gets the created event instance.
                # Third, it scatters the event to dependees who wait for
                # the current task.
                env = self.req.environment
                with _scheduler_locals._environment_scope(env), env:
                    self.events = env.get_events_from_components()
                    if len(self.dependent_events) > 0:
                        # Only wait events if any remaining dependent tasks exist.
                        # TODO(lhc): This does not support nested CPU tasks from GPU tasks.
                        #            When we notify dependees, we don't know places on
                        #            which the dependees will run since those would decide
                        #            at the next step, mapping step.
                        #            Therefore, the current runtime does not consider
                        #            dependees' placements, but just creates events
                        #            on the devices where the current task is running.
                        #            For example, when CPU tasks get GPU events,
                        #            they will just skip and will not wait for them.
                        #            This is not technical limitation, just need refactoring
                        #            and adding more features.
                        #            But our target applications do not have this pattern
                        #            and will do it later.
                        env.wait_dependent_events(self.dependent_events)
                    # Already waited all dependent events.
                    # Initialize the dependent event list.
                    self.dependent_events = []
                    write_flag = True
                    if (self._operand_type == OperandType.IN):
                        write_flag = False
                    # Move data to current device
                    dev_type = get_current_devices()[0]
                    dev_no = -1
                    if (dev_type.architecture is not cpu):
                        dev_no = dev_type.index
                    self._target_data._auto_move(device_id=dev_no, do_write=write_flag)
                    # Events could be multiple for multiple devices task.
                    env.record_events()
                    if len(self.events) > 0:
                        # If any event created by the current task exist,
                        # notify dependees and make them wait for that event,
                        # not Parla task completion.
                        self._notify_dependees_mutex(self.events)
                    env.sync_events()
                # TODO(lhc):
                #if task_state is None:
                task_state = TaskCompleted(None)
            except Exception as e:
                task_state = TaskException(e)
                logger.exception("Exception in task")
            finally:
                logger.info("Finally for task %r", self)

                # DON'T deallocate resources!
                # DataMovementTask has the same resources as the ComputeTask which created it
                # That ComputeTask will do the deallocation
                # If we do it here, we overly deallocate

                # Don't update parray tracking information either
                # The scheduler already registered the new location
                # If size changes, the ComputeTask will take care of that

                # Protect the case that it notifies dependees and
                # any dependee task is spawned before setting state.
                with self._mutex:
                    # Regardless of the previous notification,
                    # (So, before leaving the current run(), the above)
                    # it should notify dependees since
                    # new dependees could be added after the above
                    # notifications, while other devices are running
                    # their kernels asynchronously.
                    self._notify_dependees()
                    self._set_state(task_state)

        except Exception as e:
            logger.exception("Task %r: Exception in task handling", self)
            raise e


class _TaskLocals(threading.local):
    def __init__(self):
        super(_TaskLocals, self).__init__()
        self.task_scopes: List[List[ComputeTask]] = []

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
            # If its task is not yet spawned,
            # return None.
            return None
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

    @property
    def dependencies(self) -> Collection:
        return self._dependencies

    @dependencies.setter
    def dependencies(self, v: Collection):
        self._dependencies = v

    def __hash__(self):
        return hash(self._id)

    def __repr__(self):
        return "TaskID({}, task={})".format(self.full_name, self._task)

    def __str__(self):
        return "<TaskID {}>".format(self.full_name)

    def __await__(self):
        return (yield TaskAwaitTasks([self.task], self.task))


class InvalidSchedulerAccessException(RuntimeError):
    pass


class SchedulerContext(metaclass=ABCMeta):
    def spawn_task(
        self,
        function,
        args,
        deps,
        taskid,
        req,
        dataflow,
        name: Optional[str] = None,
        num_unspawned_deps: int = 0
    ):
        return ComputeTask(
            function, args, deps, taskid, req, dataflow, name,
            num_unspawned_deps
        )

    @abstractmethod
    def enqueue_task(self, Task):
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

    @abstractmethod
    def enqueue_spawned_task(self, task: Task):
        return NotImplementedError()


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


def get_current_devices() -> List[Device]:
    """
    :return: A list of `devices<parla.device.Device>` assigned to the current task. This will have one element unless `ndevices` was \
      provided when the task was `spawned<spawn>`.
    """
    return list(get_devices())


class ControllableThread(threading.Thread, metaclass=ABCMeta):
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


class WorkerThreadException(RuntimeError):
    pass


class WorkerThread(ControllableThread, SchedulerContext):
    def __init__(self, scheduler, index):
        super().__init__()
        self._monitor = threading.Condition(threading.Lock())
        self.index = index
        self._scheduler = scheduler
        self.task = None
        self._status = "Initializing"

    @property
    def scheduler(self) -> "Scheduler":
        return self._scheduler

    def incr_active_tasks(self):
        self.scheduler.incr_active_tasks()

    def decr_active_tasks(self):
        self.scheduler.decr_active_tasks()

    def enqueue_spawned_task(self, task: Task):
        self.scheduler.enqueue_spawned_task(task)

    def enqueue_task(self, task: Task):
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

    def assign_task(self, task: Task):
        with self._monitor:
            if self.task:
                raise WorkerThreadException("Tried to assign task to WorkerThread that already had one.")
            self.task = task
            self._monitor.notify()

    def _remove_task(self):
        with self._monitor:
            if not self.task:
                raise WorkerThreadException("Tried to remove a nonexistent task.")
            self.task = None

    def run(self) -> None:
        try:
            with self:
                for component in self.scheduler.components:
                    component.initialize_thread()
                while self._should_run:
                    self._status = "Getting Task"
                    if not self.task:
                        logger.debug("[%r] Blocking for a task: (%s)", self, self._monitor)
                        with self._monitor:
                            self._monitor.wait()
                        logger.debug("[WorkerThread %d] Waking up.", self.index)

                    # Thread wakes up with a task
                    if self.task:
                        logger.debug(f"[WorkerThread %d] Starting: %s", self.index, self.task.name)
                        self._status = "Running Task {}".format(self.task)
                        self.task.run()
                        self._remove_task()
                        self.scheduler.append_free_thread(self)
                    # Thread wakes up without a task (should only happen at end of program)
                    elif not self.task and self._should_run:
                        raise WorkerThreadException("%r woke up without a valid task.", self)
        except Exception as e:
            logger.exception("Unexpected exception in Task handling")
            self.scheduler.stop()

    def dump_status(self, lg=logger):
        lg.info("%r:\n%r", self, self._queue)

    def __repr__(self):
        return "<{} {} {}>".format(type(self).__name__, self.index, self._status)


# Two major TODO (ses) items:
# 1) Fix the mutexes here. There are places where if the GIL switched, stuff could break
# 2) Provide a way for the mapper to evict if the resourcepool shows a task can't be mapped anywhere and we need to clear space
class ResourcePool:
    # Importing this at the top of the file breaks due to circular dependencies
    from parla.parray.core import CPU_INDEX, PArray

    _monitor: threading.Condition
    _devices: Dict[Device, Dict[str, float]]
    _device_indices: List[int]
    _managed_parrays: Dict[int, Dict[Device, bool]]

    # Resource pools track device resources. Environments are a separate issue and are not tracked here. Instead,
    # tasks will consume resources based on their devices even though those devices are bundled into an environment.

    def __init__(self):
        self._monitor = threading.Condition(threading.Lock())

        # Devices are stored in a dict keyed by the device.
        # Each entry stores a dict with cores, memory, etc. info based on the architecture
        self._devices = self._initial_resources()

        # Parla tracks managed PArrays' locations
        # Index into dict with id(array), then with device. True means the array is present there
        # We use the unique id of the array as the key because PArray is an unhashable class
        self._managed_parrays = {}

    @staticmethod
    def _initial_resources():
        return {dev: {name: amt for name, amt in dev.resources.items()} for dev in get_all_devices()}

    ### RESOURCE ALLOCATION CALLS ###
    # These may be over-engineered, by I (Sean) haven't touched them.
    # They basically can add or reduce a resource amount (which is just memory right now)

    # Currently, resource allocation is done when the task is MAPPED.
    # The task itself does not allocate.
    # By the time it starts running, resources have already been allocated for it.
    def allocate_resources(self, d: Device, resources: ResourceDict, *, blocking: bool = False) -> bool:
        """Allocate the resources described by `dd`.

        :param d: The device on which resources exist.
        :param resources: The resources to allocate.
        :param blocking: If True, this call will block until the resource is available and will always return True.

        :return: True iff the allocation was successful.
        """
        return self._atomically_update_resources(d, resources, -1, blocking)

    # Currently, resource deallocation is done when the task finishes.
    def deallocate_resources(self, d: Device, resources: ResourceDict) -> None:
        """Deallocate the resources described by `dd`.

        :param d: The device on which resources exist.
        :param resources: The resources to deallocate.
        """
        ret = self._atomically_update_resources(d, resources, 1, False)
        assert ret

    def check_resources_availability(self, d: Device, resources: ResourceDict):
        """Check if necessary resouces of the deviced is available.

        :param d: The device on which resources exist.
        :param resources: The resources to deallocate.
        """
        with self._monitor:
            is_available = True
            for name, amount in resources.items():
                dres = self._devices[d]

                # Workaround stupid vcus (I'm getting rid of these at some point)
                if d.architecture.id == 'gpu' and name == 'vcus':
                    continue

                if amount > dres[name]:
                    is_available = False
                logger.debug("Resource check for %d %s on device %r: %s", amount, name, d, "Passed" if is_available else "Failed")
            return is_available

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

            logger.info("[ResourcePool] Attempted to allocate %s * %r (blocking %s) => %s", \
                         multiplier, (d, resources), block, "success" if success else "fail")
            if to_release:
                logger.info("Releasing resources due to failure: %r", to_release)

            for name, v in to_release:
                ret = self._update_resource(d, name, -v * multiplier, block)
                assert ret

            assert not success or len(to_release) == 0  # success implies to_release empty
            return success

    def _update_resource(self, dev: Device, res: str, amount: float, block: bool):
        # Workaround stupid vcus (I'm getting rid of these at some point)
        if dev.architecture.id == 'gpu' and res == 'vcus':
            return True
        try:
            while True:  # contains return
                dres = self._devices[dev]
                if -amount <= dres[res]:
                    dres[res] += amount
                    if amount > 0:
                        self._monitor.notify_all()
                    assert dres[res] <= dev.resources[res], "{}.{} was over deallocated".format(dev, res)
                    assert dres[res] >= 0, "{}.{} was over allocated".format(dev, res)
                    return True
                else:
                    if block:
                        self._monitor.wait()
                    else:
                        return False
        except KeyError:
            raise ValueError("Resource {}.{} does not exist".format(dev, res))

    ### PARRAY MEMORY TRACKING CALLS ###

    # parrays don't use devices, they use indices
    # CPU is CPU_INDEX, GPUs are positive integers
    # This helper function translates from a Device class to its PArray ID quickly
    def _to_parray_index(self, device):
        if device.architecture == cpu:
            return self.CPU_INDEX
        if device.architecture.id == 'gpu':
            return device.index
        raise NotImplementedError("Only cpu and gpu architectures are supported")

    # Start tracking the memory usage of a parray
    def track_parray(self, parray):
        logger.debug(f"[ResourcePool] Tracking parray with ID %d in these locations:", id(parray))
        # Figure out all the locations where a parray exists
        parray_location_map = {}
        for device in self._devices:
            device_id = self._to_parray_index(device)
            if parray.exists_on_device(device_id):
                parray_location_map[device] = True

                logger.debug(f"[ResourcePool]   - %r", device)

                # Update the resource usage at this location
                self.allocate_resources(device, {'memory': parray.nbytes})
            else:
                parray_location_map[device] = False

        # Insert the location map into our dict, keyed by the parray itself
        with self._monitor:
            self._managed_parrays[id(parray)] = parray_location_map

    # Stop tracking the memory usage of a parray
    def untrack_parray(self, parray):
        logger.debug(f"[ResourcePool] Untracking parray with ID %d from these locations:", id(parray))
        # Return resources to the devices
        for device, parray_exists in self._managed_parrays[id(parray)].items():
            if parray_exists:
                self.deallocate_resources(device, {'memory': parray.nbytes})
                logger.debug(f"[ResourcePool]   - %r", device)

        # Delete the dictionary entry
        with self._monitor:
            del self._managed_parrays[id(parray)]

    # Notify the resource pool that a device has a new instantiation of an array
    def add_parray_to_device(self, parray, device):
        logger.debug(f"[ResourcePool] Adding parray with ID %d to device %r", id(parray), device)
        if self._managed_parrays[id(parray)][device] == True:
            #raise ValueError("Tried to register a parray on a device where it already existed")
            logger.debug(f"[ResourcePool]   (It was already there...)")
            return
        with self._monitor:
            self._managed_parrays[id(parray)][device] = True
        self.allocate_resources(device, {'memory': parray.nbytes})

    # Notify the resource pool that an instantiation of an array has been deleted
    def remove_parray_from_device(self, parray, device):
        logger.debug(f"[ResourcePool] Removing parray with ID %d from device %r", id(parray), device)
        if self._managed_parrays[id(parray)][device] == False:
            #raise ValueError("Tried to remove a parray from a device where it didn't exist")
            logger.debug(f"[ResourcePool]   (It wasn't there...)")
            return
        with self._monitor:
            self._managed_parrays[id(parray)][device] = False
        self.deallocate_resources(device, {'memory': parray.nbytes})

    # On a parray move, call this to start tracking the parray (if necessary) and update its location
    def register_parray_move(self, parray, device):
        if id(parray) not in self._managed_parrays:
            self.track_parray(parray)
            # If this new array originates on the dest device, skip the next step
            if self._managed_parrays[id(parray)][device]:
                return
        self.add_parray_to_device(parray, device)

    def parray_is_on_device(self, parray, device):
        with self._monitor:
            return (id(parray) in self._managed_parrays) and (self._managed_parrays[id(parray)][device])

    def __repr__(self):
        return "ResourcePool(devices={})".format(self._devices)

    def get_resources(self):
        return [dev for dev in self._devices]


class AssignmentFailed(Exception):
    pass


_T = TypeVar('_T')


def shuffled(lst: Iterable[_T]) -> List[_T]:
    """Shuffle a list non-destructively."""
    lst = list(lst)
    random.shuffle(lst)
    return lst


class Scheduler(ControllableThread, SchedulerContext):
    # See __init__ function below for comments on the functionality of these members
    _environments: TaskEnvironmentRegistry
    _worker_threads: List[WorkerThread]
    _free_worker_threads: Deque[WorkerThread]
    _available_resources: ResourcePool
    _device_compute_task_counts: Dict[Device, int]
    period: float

    def __init__(self, environments: Collection[TaskEnvironment], n_threads: int = None, period: float = 1.4012985e-20):
        # ControllableThread: __init__ sets it to run
        # SchedulerContext: No __init__
        super().__init__()

        # TODO(lhc): for now, assume that n_threads is always None.
        #            Each device needs a dedicated thread.
        n_threads = sum(d.resources.get("vcus", 1) for e in environments for d in e.placement)

        self._environments = TaskEnvironmentRegistry(*environments)

        # Empty list for storing reported exceptions at runtime
        self._exceptions = []

        # Start with one count that is removed when the scheduler is "exited"
        self._active_task_count = 1

        # Period scheduler sleeps between loops (see run function)
        self.period = period

        self._monitor = threading.Condition(threading.Lock())

        # Track, allocate, and deallocate resources (devices)
        self._available_resources = ResourcePool()

        # Spawned task queues
        # Tasks that have been spawned but not mapped are stored here.
        # Tasks are removed once they are mapped.
        # Spawned queue consists of two levels, current and new.
        # Newly spawned tasks or tasks which fail to schedule are always
        # enqueued on the "new" queue.
        # When the mapper runs, it moves all tasks from the "new" to the "current" queue.
        # Tasks which the mapper will try to map at the current
        # iteration are always dequeued from the current queue.
        # This implementation is simple and avoids a long-running mapper in the case where new
        # tasks spawn as it runs
        self._spawned_task_queue = deque()
        self._new_spawned_task_queue = deque()

        # This is where tasks go when they have been mapped and their
        # dependencies are complete, but they have not been scheduled.
        self._ready_queue = deque()

        # The device queues where scheduled tasks go to be launched from
        self._device_queues = {dev: deque() for dev in self._available_resources.get_resources()}

        # The number of in-flight compute tasks on each device
        self._device_compute_task_counts = {dev: 0 for dev in self._available_resources.get_resources()}

        # Dictinary mapping data block to task lists.
        self._datablock_dict = defaultdict(list)

        self._worker_threads = [WorkerThread(self, i) for i in range(n_threads)]
        for t in self._worker_threads:
            t.start()
        self._free_worker_threads = deque(self._worker_threads)
        # Start the scheduler thread (likely to change later)
        self.start()

    @property
    @lru_cache(maxsize=1)
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
            #t.stop() # This is needed to gracefully end the threads without throwing missing task exceptions
            t.join()  # This is what actually rejoins the threads
        if self._exceptions:
            # TODO: Should combine all of them into a single exception.
            raise self._exceptions[0]

    def append_free_thread(self, thread: WorkerThread):
        with self._monitor:
            self._free_worker_threads.append(thread)

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
            try:
                task = self._spawned_task_queue.pop()
                logger.debug(f"[Scheduler] Popped %r from spawn queue.", task)
                return task
            except IndexError:
                return None

    def enqueue_task(self, task: Task):
        """Enqueue a task on the resource allocation queue.
           Note that this enqueue has no data race.
        """
        self._ready_queue.appendleft(task)

    def _dequeue_task(self, timeout=None) -> Optional[Task]:
        """Dequeue a task from the resource allocation queue.
        """
        while True:
            try:
                if self._should_run:
                    task = self._ready_queue.pop()
                    logger.debug(f"[Scheduler] Popped %r from ready queue.", task)
                    return task
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
        logger.debug(f"[Scheduler] Mapping %r.", task)

        # Sean: The goal of the mapper look at data locality and load balancing
        # and pick a suitable set of devices on which to run a task.
        # Currently, it just supports single-device tasks (like everything else...)
        # Tasks have a set of requirements passed to them by @spawn. We need to
        # match those requirements and find the most suitable device.
        possible_devices = task.req.devices
        max_suitability = None
        best_device = None
        best_device_owns_dependency = False
        best_device_local_data = 0
        for device in possible_devices:
            # Ensure that the device has enough resources for the task
            if not self._available_resources.check_resources_availability(device, task.req.resources):
                continue

            # THIS IS THE MEAT OF THE MAPPING POLICY
            # We calculate a few constants based on data locality and load balancing
            # We then add those together with tunable weights to determine a suitability
            # The device with the highest suitability is the lucky winner

            # First, we calculate data on the device and data to be moved to the device
            local_data = 0
            nonlocal_data = 0
            for parray in task.dataflow.input + task.dataflow.inout:
                if self._available_resources.parray_is_on_device(parray, device):
                    local_data += parray.nbytes
                else:
                    nonlocal_data += parray.nbytes

            # These values are really big, so I'm normalizing them to the size of the
            # device memory so my monkey brain can fathom the numbers
            local_data /= device.resources['memory']
            nonlocal_data /= device.resources['memory']

            # Figure out whether the task has a dependency running on this device
            this_device_owns_dependency = False
            for dependency in task.dependencies:
                if device in dependency.req.devices:
                    this_device_owns_dependency = True
                    break

            # If the best device owns a dependency, but this one doesn't,
            # AND the best device has more local data, we can skip this device
            # TODO (ses): Ignore this if the dependency has finished running!
            if best_device_owns_dependency and not this_device_owns_dependency and best_device_local_data >= local_data:
                continue

            # Next we calculate the load-balancing factor
            # For now this is just a count of tasks on the device queue (TODO (ses): better heuristics later...)
            dev_load = self._device_compute_task_counts[device]

            # Normalize this too so we have numbers between 0 and 1
            dev_load /= self._active_task_count

            # TODO (ses): Move these magic numbers somewhere better
            local_data_weight = 30.0
            nonlocal_data_weight = 10.0
            load_weight = 1.0

            # Calculate the suitability
            suitability = local_data_weight * local_data \
                        - nonlocal_data_weight * nonlocal_data \
                        - load_weight * dev_load
            """
            def myformat(num):
                return "{:.3f}".format(num)

            print(f"dev={device}", end='   ')
            print(f"dep={this_device_owns_dependency}", end='   ')
            print(f"local={myformat(local_data)}", end='   ')
            print(f"nonlocal={myformat(nonlocal_data)}", end='   ')
            print(f"load={myformat(dev_load)}", end='   ')
            print(f"suit={myformat(suitability)}")
            """

            # Update whether or not this is the most suitable device
            if max_suitability is None or suitability > max_suitability:
                max_suitability = suitability
                best_device = device
                best_device_owns_dependency = this_device_owns_dependency
                best_device_local_data = local_data

        if best_device is None:
            logger.debug(f"[Scheduler] Failed to map %r.", task)
            return False

        # Stick this info in an environment (I based this code on the commented out stuff below)
        task_env_gen = self._environments.find_all(placement={best_device}, tags={}, exact=True)
        task_env = next(task_env_gen)
        task.req = EnvironmentRequirements(task.req.resources, task_env, task.req.tags)

        task.set_assigned()
        logger.debug(f"[Scheduler] Mapped %r.", task)
        return True

        # Sean: I don't understand the old way. Just commenting it out and acting like it doesn't exist.
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
                logger.debug(f"[Scheduler] Mapped %r.", task)
                return True
        logger.debug(f"[Scheduler] Failed to map %r.", task)
        return False
        """

    def fill_curr_spawned_task_queue(self):
        """ It moves tasks on the new spawned task queue to
            the current queue.
        """
        with self._monitor:
            if (len(self._new_spawned_task_queue) > 0):
                new_q = self._new_spawned_task_queue
                new_tasks = [new_q.popleft() for _ in range(len(new_q))]
                # Newly added tasks should be enqueued onto the
                # right to guarantee FIFO manners.
                # It is efficient to map higher priority tasks to devices
                # first since Applications generally spawn
                # tasks in priority orders.
                self._spawned_task_queue.extend(new_tasks)

    def fill_curr_mapped_task_queue(self):
        """ It moves tasks on the new mapped task queue to
            the current queue.
        """
        with self._monitor:
            new_q = self._new_mapped_task_queue
            new_tasks = [new_q.popleft() for _ in range(len(new_q))]
            if len(new_tasks) > 0:
                self._mapped_task_queue.extendleft(new_tasks)

    def _construct_datamove_task(self, target_data, compute_task: ComputeTask, operand_type: OperandType):
        """
          This function constructs data movement task for target data.
          This function consists of two steps.
          First, it iterates all operand data of the dependency tasks
          of the computation task (original task).
          If any of the dependency tasks' data is overlapped with the
          target data, then add the dependency task to the new data
          movement task's dependency list.
          Second, construct a data movement task.
        """
        # Construct data movement task.
        taskid = TaskID(str(compute_task.taskid) + "." + str(hex(id(target_data))) + ".dmt." + str(len(task_locals.global_tasks)), (len(task_locals.global_tasks), ))
        task_locals.global_tasks += [taskid]
        datamove_task = DataMovementTask(compute_task, taskid,
                                         compute_task.req, target_data, operand_type,
                                         str(compute_task.taskid) + "."
                                        + str(hex(id(target_data))) + ".dmt")
        self.incr_active_tasks()
        compute_task._add_dependency_mutex(datamove_task)
        target_data_id = id(target_data)
        is_overlapped = False
        if target_data_id in self._datablock_dict:
            # Get task lists using the target data block.
            dep_task_list = self._datablock_dict[target_data_id]
            completed_tasks = []
            for dep_task_tuple in dep_task_list:
                dep_task_id = dep_task_tuple[0]
                dep_task = dep_task_tuple[1]
                # Only checks dependent tasks if they use the same data blocks.
                if compute_task.is_dependent(dep_task):
                    if not datamove_task._add_dependency(dep_task):
                        completed_tasks.append(dep_task_id)
            dep_task_list = [tuple(dt for dt in dep_task_list if dt[0] != ft) for ft in completed_tasks]
        self._datablock_dict[target_data_id].append((str(compute_task.taskid), compute_task))
        # If a task has no dependency after it is assigned to devices,
        # immediately enqueue a corresponding data movement task to
        # the ready queue.
        if not datamove_task.bool_check_remaining_dependencies():
            self.enqueue_task(datamove_task)

    def _map_tasks(self):
        # The first loop iterates a spawned task queue
        # and constructs a mapped task subgrpah.
        #logger.debug("[Scheduler] Map Phase")
        self.fill_curr_spawned_task_queue()
        while True:
            task: Optional[Task] = self._dequeue_spawned_task()
            if task:
                if not task.assigned:
                    is_assigned = self._assignment_policy(task)  # This is what actually maps the task
                    assert isinstance(is_assigned, bool)
                    if not is_assigned:
                        self.enqueue_spawned_task(task)
                    else:
                        # Create data movement tasks for each data
                        # operands of this task.
                        # TODO(lhc): this is not good.
                        #            will use logical values to make it easy to understand.
                        for data in task.dataflow.input:
                            self._construct_datamove_task(data, task, OperandType.IN)
                        for data in task.dataflow.output:
                            self._construct_datamove_task(data, task, OperandType.OUT)
                        for data in task.dataflow.inout:
                            self._construct_datamove_task(data, task, OperandType.INOUT)

                        # Update parray tracking and task count on the device
                        for parray in (task.dataflow.input + task.dataflow.inout + task.dataflow.output):
                            if len(task.req.environment.placement) > 1:
                                raise NotImplementedError("Multidevice not supported")
                            for device in task.req.environment.placement:
                                self._available_resources.register_parray_move(parray, device)
                                self._device_compute_task_counts[device] += 1

                        # Allocate additional resources used by this task (blocking)
                        for device in task.req.devices:
                            for resource, amount in task.req.resources.items():
                                logger.debug("Task %r allocating %d %s on device %r", task, amount, resource, device)
                            self._available_resources.allocate_resources(device, task.req.resources, blocking=True)

                        # Only computation needs to set a assigned flag.
                        # Data movement task is set as assigned when it is created.
                        task.set_assigned()
                        # If a task has no dependency after it is assigned to devices,
                        # immediately enqueue a corresponding data movement task to
                        # the ready queue.
                        if not task.bool_check_remaining_dependencies():
                            self.enqueue_task(task)
                            logger.debug(f"[Scheduler] Enqueued %r on ready queue", task)
                else:
                    logger.exception("[Scheduler] Tasks on the spawned queue ", \
                                     "should be not assigned any device.")
                    self.stop()
            else:
                # If there is no spawned task at this moment,
                # move to the mapped task scheduling.
                break

    def _schedule_tasks(self):
        """ Currently this doesn't do any intelligent scheduling (ordering).
            Dequeue all ready tasks and send them to device queues in order.
        """
        #logger.debug("[Scheduler] Schedule Phase")
        while True:
            task: Optional[Task] = self._dequeue_task()
            if not task:
                break
            if not task.assigned:
                logger.debug("[Scheduler] Task %r: Failed to assign", task)
                break
            for d in task.req.devices:
                logger.info(f"[Scheduler] Enqueuing %r to device %r", task, d)
                self._device_queues[d].append(task)

    # _launch_[DEVICE TYPES]_tasks launches a task by assigning it to available
    # worker threads. It manages different execution paths based on the target
    # devices' types.
    # For example, in the future, the runtime will allow two task colocations
    # of computation, moving data in, and moving data out.
    # TODO(lhc): for now, the CPU/GPU launchers are same.

    def _launch_cpu_task(self, queue, task: Task, dev: Device):
        if self._available_resources.check_resources_availability(dev, task.req.resources):
            worker = self._free_worker_threads.pop()  # grab a worker
            logger.info(f"[Scheduler] Launching CPU task, %r on %r",
                        task, worker)
            # Assign the task to the worker (this notifies the worker's monitor)
            worker.assign_task(task)
            logger.debug(f"[Scheduler] Launched %r", task)
        else:
            queue.appendleft(task)

    def _launch_gpu_task(self, queue, task: Task, dev: Device):
        if self._available_resources.check_resources_availability(dev, task.req.resources):
            worker = self._free_worker_threads.pop()  # grab a worker
            logger.info(f"[Scheduler] Launching GPU task, %r on %r",
                        task, worker)
            # Assign the task to the worker (this notifies the worker's monitor)
            worker.assign_task(task)
            logger.debug(f"[Scheduler] Launched %r", task)
        else:
            queue.appendleft(task)

    def _launch_tasks(self):
        """ Iterate through free devices and launch tasks on them
        """
        #logger.debug("[Scheduler] Launch Phase")
        with self._monitor:
            for dev, queue in self._device_queues.items():
                # Make sure there's an available WorkerThread
                if len(self._free_worker_threads) == 0:
                    break
                if len(queue) > 0:  # If there are tasks on the queue.
                    try:
                        task = queue.pop()  # Grab a task.
                        if dev.architecture.id == 'cpu':
                            self._launch_cpu_task(queue, task, dev)
                        elif dev.architecture.id == 'gpu':
                            self._launch_gpu_task(queue, task, dev)
                        else:
                            raise Exception("Unsupported architecture")
                    finally:
                        pass

    def run(self) -> None:
        # noinspection PyBroadException
        try:  # Catch all exception to report them usefully
            i = 0
            while self._should_run:
                self._map_tasks()
                self._schedule_tasks()
                self._launch_tasks()
                #logger.debug("[Scheduler] Sleeping!")
                time.sleep(self.period)
                #logger.debug("[Scheduler] Awake!")

        except Exception:
            logger.exception("Unexpected exception in Scheduler")
            self.stop()

    def stop(self):
        super().stop()
        for w in self._worker_threads:
            w.stop()

    def report_exception(self, e: BaseException):
        logger.exception("Report exception:", e)
        self._exceptions.append(e)

    def dump_status(self, lg=logger):
        lg.info("%r:\n%r\navailable: %r", self,
                self._ready_queue, self._available_resources)
        w: WorkerThread
        for w in self._worker_threads:
            w.dump_status(lg)
