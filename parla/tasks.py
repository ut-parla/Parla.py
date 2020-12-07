"""
Parla supports simple task parallelism.

.. testsetup::

    T0 = None
    code = None
    from .cpu import cpu

"""
import logging
import threading
import inspect
import dis
import warnings
from abc import abstractmethod, ABCMeta
from contextlib import asynccontextmanager
from typing import Awaitable, Collection, Iterable, Optional, Any, Union, List, FrozenSet, Dict

from parla.device import Device, Architecture, get_all_devices
from parla.task_runtime import TaskCompleted, TaskRunning, TaskAwaitTasks, TaskState, DeviceSetRequirements, Task

try:
    from parla import task_runtime, array
except ImportError as e:
    # Ignore the exception if the stack includes the doc generator
    if all("sphinx" not in f.filename for f in inspect.getouterframes(inspect.currentframe())):
        raise

logger = logging.getLogger(__name__)

__all__ = [
    "TaskID", "TaskSpace", "spawn", "get_current_devices", "tasks", "finish", "CompletedTaskSpace", "Task"
]


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


class TaskSet(Awaitable, Collection, metaclass=ABCMeta):
    """
    A collection of tasks.
    """

    @property
    @abstractmethod
    def _tasks(self) -> Collection:
        pass

    @property
    def _flat_tasks(self) -> Collection:
        # Compute the flat dependency set (including unwrapping TaskID objects)
        deps = []
        for ds in self._tasks:
            if not isinstance(ds, Iterable):
                ds = (ds,)
            for d in ds:
                if hasattr(d, "task"):
                    d = d.task
                if not isinstance(d, task_runtime.Task):
                    raise TypeError("Dependencies must be TaskIDs or Tasks: " + str(d))
                deps.append(d)
        return tuple(deps)

    def __await__(self):
        return (yield TaskAwaitTasks(self._flat_tasks, None))

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks)

    def __contains__(self, x) -> bool:
        return x in self._tasks

    def __repr__(self):
        return "tasks({})".format(self._tasks)


class tasks(TaskSet):
    """
    An ad-hoc collection of tasks.
    An instance is basically a reified dependency list as would be passed to `spawn`.
    This object is awaitable and will block until all tasks are complete.

    >>> await tasks(T1, T2)
    >>> @spawn(None, tasks(T1, T2)) # Same as @spawn(None, [T1, T2])
    >>> def f():
    >>>     pass
    """

    @property
    def _tasks(self) -> Collection:
        return self.args

    __slots__ = ("args",)

    def __init__(self, *args):
        self.args = args


class TaskSpace(TaskSet):
    """A collection of tasks with IDs.

    A `TaskSpace` can be indexed using any hashable values and any
    number of "dimensions" (indicies). If a dimension is indexed with
    numbers then that dimension can be sliced.

    >>> T = TaskSpace()
    ... for i in range(10):
    ...     @spawn(T[i], [T[0:i-1]])
    ...     def t():
    ...         code

    This will produce a series of tasks where each depends on all previous tasks.

    :note: `TaskSpace` does not support assignment to indicies.
    """
    _data: Dict[int, TaskID]

    @property
    def _tasks(self):
        return self._data.values()

    def __init__(self, name="", members=None):
        """Create an empty TaskSpace.
        """
        self._name = name
        self._data = members or {}

    def __getitem__(self, index):
        """Get the `TaskID` associated with the provided indicies.
        """
        if not isinstance(index, tuple):
            index = (index,)
        ret = []

        def traverse(prefix, index):
            if len(index) > 0:
                i, *rest = index
                if isinstance(i, slice):
                    for v in range(i.start or 0, i.stop, i.step or 1):
                        traverse(prefix + (v,), rest)
                elif isinstance(i, Iterable):
                    for v in i:
                        traverse(prefix + (v,), rest)
                else:
                    traverse(prefix + (i,), rest)
            else:
                ret.append(self._data.setdefault(prefix, TaskID(self._name, prefix)))

        traverse((), index)
        # print(index, ret)
        if len(ret) == 1:
            return ret[0]
        return ret

    def __repr__(self):
        return "TaskSpace({_name}, {_data})".format(**self.__dict__)


class CompletedTaskSpace(TaskSet):
    """
    A task space that returns completed tasks instead of unused tasks.

    This is useful as the base case for more complex collections of tasks.
    """

    @property
    def _tasks(self) -> Collection:
        return []

    def __getitem__(self, index):
        return tasks()


def get_placement_for_value(p: Union[Architecture, Device, Task, TaskID, Any]) -> List[Device]:
    if hasattr(p, "__parla_placement__"):
        # this handles Architecture, ResourceRequirements, and other types with __parla_placement__
        return list(p.__parla_placement__())
    elif isinstance(p, Device):
        return [p]
    elif isinstance(p, TaskID):
        return get_placement_for_value(p.task)
    elif isinstance(p, task_runtime.Task):
        return get_placement_for_value(p.req)
    elif array.is_array(p):
        return [array.get_memory(p).device]
    elif isinstance(p, Collection):
        raise TypeError("Collection passed to get_placement_for_value, probably needed get_placement_for_set: {}"
                        .format(type(p)))
    else:
        raise TypeError(type(p))


def get_placement_for_set(placement: Collection[Union[Architecture, Device, Task, TaskID, Any]]) -> FrozenSet[Device]:
    if not isinstance(placement, Collection):
        raise TypeError(type(placement))
    return frozenset(d for p in placement for d in get_placement_for_value(p))


def get_placement_for_any(placement: Union[Collection[Union[Architecture, Device, Task, "TaskID", Any]], Any, None]) \
        -> FrozenSet[Device]:
    if placement is not None:
        ps = placement if isinstance(placement, Iterable) and not array.is_array(placement) else [placement]
        return get_placement_for_set(ps)
    else:
        return frozenset(get_all_devices())


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


_task_locals = _TaskLocals()


def detect_data(body):

    var_maps = {}
    array_names = set()
    int_names = set()
    #Locate all variable related to data stream
    if not body.__globals__ == None:
        for key, val in body.__globals__.items():
            if array.is_array(val) or isinstance(val, list):
                var_maps[key] = val
                array_names.add(key)
            elif isinstance(val,int):
                var_maps[key] = val
                int_names.add(key)
    #local all nonlcoals
    code = body.__code__
    n_nonlocals = len(code.co_freevars)
    for i in range(n_nonlocals):
        cur_name = code.co_freevars[i]
        cur_val = body.__closure__[i].cell_contents
        var_maps[cur_name] = cur_val
        if array.is_array(cur_val) or isinstance(cur_val, list):
            array_names.add(cur_name)
        elif isinstance(cur_val,int):
            int_names.add(cur_name)


    idx_maps = {}
    cur_segments= []
    cur_arr = None
    read = {}
    write = {}

    for ins in dis.get_instructions(code):
        #load sequence
        if ins.opcode == 136 or ins.opcode == 116:
            if ins.argrepr in array_names:
                if not cur_arr == None:
                    read[cur_arr] = None
                cur_arr = ins.argrepr
            elif ins.argrepr in int_names:
                cur_segments.append(var_maps[ins.argrepr])
        #store sequence
        if ins.opcode == 137 or ins.opcode == 97:
            if ins.argrepr in array_names:
                write[ins.argrepr] = None
                cur_arr = None
        if cur_arr is not None:
            #load const
            if ins.opcode == 100:
                #we only care about int in index
                if  ins.argrepr.isdigit():
                    cur_segments.append(int(ins.argrepr))
                    #const.append(int(ins.argrepr))
            #binary power
            elif ins.opcode == 19:
                add = cur_segments[-2:]
                results = add[0]**add[1]
                cur_segments = cur_segments[:-2]+[results]
            #binary multiply
            elif ins.opcode ==20:
                add = cur_segments[-2:]
                results = add[0]*add[1]
                cur_segments = cur_segments[:-2]+[results]
            #binary mod
            elif ins.opcode ==22:
                add = cur_segments[-2:]
                results = add[0]%add[1]
                cur_segments = cur_segments[:-2]+[results]
            #binary add
            elif ins.opcode ==23:
                add = cur_segments[-2:]
                results = add[0]+add[1]
                cur_segments = cur_segments[:-2]+[results]
            #binary sub
            elif ins.opcode ==24:
                add = cur_segments[-2:]
                results = add[0]-add[1]
                cur_segments = cur_segments[:-2]+[results]
            #binary floor divide
            elif ins.opcode ==26:
                add = cur_segments[-2:]
                results = add[0]//add[1]
                cur_segments = cur_segments[:-2]+[results]
            #build index tuple
            elif ins.opcode == 102:
                index = tuple(cur_segments)
                cur_segments = [index]
            #build index slice
            elif ins.opcode == 133:
                index = slice(*cur_segments)
                cur_segments = [index]
            #binary subscribe
            elif ins.opcode == 25:
                idx = cur_segments[-1]
                if cur_arr in read.keys() and not read[cur_arr]==None:
                    read[cur_arr].append(idx)
                else:
                    read[cur_arr]=[idx]
                cur_segments = cur_segments[:-1]
                cur_arr = None
            #store subs
            elif ins.opcode == 60:
                idx = cur_segments[-1]
                if cur_arr in write.keys():
                    write[cur_arr].append(idx)
                else:
                    write[cur_arr]=[idx]
                cur_segments = cur_segments[:-1]
                cur_arr = None

    if not cur_arr == None:
        read[cur_arr] = None

    data = []
    for key,vals in read.items():
        cur_data = var_maps[key]
        if vals is not None and len(vals) > 0:
            for val in vals:
                data.append(cur_data[val])
        else:
            data.append(cur_data)
    if data == []:
        return None

    return data


def _move_function_local(body):
    """
    A function copy all data to desired device
    """
    new_global = {}
    new_closure= []



    if not body.__globals__ == None:
        for key, val in body.__globals__.items():
            if array.is_array(val) or isinstance(val, list):
                local_array = array.get_device_array(val)
                new_global[key]=local_array
            else:
                new_global[key] = val


    if not body.__closure__ == None:
        for x in body.__closure__:
            val = x.cell_contents
            if array.is_array(val) or isinstance(val, list):
                local_array = array.get_device_array(val)
                new_cell = _make_cell(local_array)
            else:
                new_cell = x
            new_closure.append(new_cell)


    new_body = type(body)(
            body.__code__, new_global, body.__name__, body.__defaults__,
            closure=tuple(new_closure))
    new_body.__annotations__ = body.__annotations__
    new_body.__doc__ = body.__doc__
    new_body.__kwdefaults__ = body.__kwdefaults__
    new_body.__module__ = body.__module__
    return new_body


def _task_callback(task, body) -> TaskState:
    """
    A function which forwards to a python function in the appropriate device context.
    """
    try:
        body = body

        if inspect.isfunction(body):
            body = _move_function_local(body)

        if inspect.iscoroutinefunction(body):
            logger.debug("Constructing coroutine task: %s", task.taskid)
            body = body()

        if inspect.iscoroutine(body):
            try:
                in_value_task = getattr(task, "value_task", None)
                in_value = in_value_task and in_value_task.result
                logger.debug("Executing coroutine task: %s with input %s from %r", task.taskid,
                             in_value_task, in_value)
                new_task_info = body.send(in_value)
                task.value_task = None
                if not isinstance(new_task_info, TaskAwaitTasks):
                    raise TypeError("Parla coroutine tasks must yield a TaskAwaitTasks")
                dependencies = new_task_info.dependencies
                value_task = new_task_info.value_task
                if value_task:
                    assert isinstance(value_task, task_runtime.Task)
                    task.value_task = value_task
                return TaskRunning(_task_callback, (body,), dependencies)
            except StopIteration as e:
                result = None
                if e.args:
                    (result,) = e.args
                return TaskCompleted(result)
        else:
            logger.debug("Executing function task: %s", task.taskid)
            result = body()
            return TaskCompleted(result)
    finally:
        logger.debug("Finished: %s", task.taskid)
    assert False


def _make_cell(val):
    """
    Create a new Python closure cell object.

    You should not be using this. I shouldn't be either, but I don't know a way around Python's broken semantics.
    """
    x = val

    def closure():
        return x

    return closure.__closure__[0]


def spawn(taskid: Optional[TaskID] = None, dependencies = (), *,
          memory: int = None,
          vcus: float = None,
          placement: Union[Collection[Union[Architecture, Device, Task, TaskID, Any]], Any] = None,
          ndevices: int = 1,
          tags: Collection[Any] = (),
          data: Collection[Any] = None
          ):
    """
    spawn(taskid: Optional[TaskID] = None, dependencies = (), *, memory: int = None, placement: Collection[Any] = None, ndevices: int = 1)

    Execute the body of the function as a new task. The task may start
    executing immediately, so it may execute in parallel with any
    following code.

    >>> @spawn(T1, [T0]) # Create task with ID T1 and dependency on T0
    ... def t():
    ...     code

    >>> @spawn(T1, [T0], placement=cpu)
    ... def t():
    ...     code

    :param taskid: the ID of the task in a `TaskSpace` or None if the task does not have an ID.
    :param dependencies: any number of dependency arguments which may be `Tasks<Task>`, `TaskIDs<TaskID>`, or \
       iterables of Tasks or TaskIDs.
    :param memory: The amount of memory this task uses.
    :param placement: A collection of values (`~parla.device.Architecture`, `~parla.device.Device`, or array data) which \
       specify devices at which the task can be placed.
    :param ndevices: The number of devices the task will use. If `ndevices` is greater than 1, the `memory` is divided \
       evenly between the devices. In the task: `len(get_current_devices()) == ndevices<get_current_devices>`.

    The declared task (`t` above) can be used as a dependency for later tasks (in place of the tasks ID).
    This same value is stored into the task space used in `taskid`.

    :see: :ref:`Fox's Algorithm` Example

    """
    # :param vcus: The amount of compute power this task uses. It is specified in "Virtual Compute Units".
    # TODO: Document tags argument

    if not taskid:
        taskid = TaskID("global_" + str(len(_task_locals.global_tasks)), (len(_task_locals.global_tasks),))
        _task_locals.global_tasks += [taskid]

    def decorator(body):
        nonlocal placement, memory

        #inspect_data,_ = detect_data(body)
        inspect_data = None

        if data is not None:
            if placement is not None or memory is not None:
                raise ValueError("The data parameter cannot be combined with placement or memory paramters.")
            placement = data
            memory = array.storage_size(*data)
        elif inspect_data is not None:
            warnings.warn("data detected but not declare")
            if placement is not None or memory is not None:
                raise ValueError("The data parameter cannot be combined with placement or memory paramters.")
            placement = inspect_data
            memory = array.storage_size(*inspect_data)


        devices = get_placement_for_any(placement)

        resources = {}
        if memory is not None:
            resources["memory"] = memory
        if vcus is not None:
            resources["vcus"] = vcus

        req = DeviceSetRequirements(resources, ndevices, devices, tags)

        if inspect.isgeneratorfunction(body):
            raise TypeError("Spawned tasks must be normal functions or coroutines; not generators.")

        # Compute the flat dependency set (including unwrapping TaskID objects)
        deps = tasks(*dependencies)._flat_tasks

        if inspect.iscoroutine(body):
            # An already running coroutine does not need changes since we assume
            # it was changed correctly when the original function was spawned.
            separated_body = body
        else:
            # Perform a horrifying hack to build a new function which will
            # not be able to observe changes in the original cells in the
            # tasks outer scope. To do this we build a new function with a
            # replaced closure which contains new cells.
            separated_body = type(body)(
                body.__code__, body.__globals__, body.__name__, body.__defaults__,
                closure=body.__closure__ and tuple(_make_cell(x.cell_contents) for x in body.__closure__))
            separated_body.__annotations__ = body.__annotations__
            separated_body.__doc__ = body.__doc__
            separated_body.__kwdefaults__ = body.__kwdefaults__
            separated_body.__module__ = body.__module__

        taskid.dependencies = dependencies

        # Spawn the task via the Parla runtime API
        task = task_runtime.get_scheduler_context().spawn_task(
            _task_callback, (separated_body,), deps, taskid=taskid, req=req)

        logger.debug("Created: %s %r", taskid, body)

        for scope in _task_locals.task_scopes:
            scope.append(task)

        # Return the task object
        return task

    return decorator


def get_current_devices() -> List[Device]:
    """
    :return: A list of `devices<parla.device.Device>` assigned to the current task. This will have one element unless `ndevices` was \
      provided when the task was `spawned<spawn>`.
    """
    return list(task_runtime.get_devices())


@asynccontextmanager
async def finish():
    """
    Execute the body of the `with` normally and then perform a barrier applying to all tasks created within this block
    and in this task.

    `finish` does not wait for tasks which are created by the tasks it waits on. This is because tasks are allowed to
    complete before tasks they create. This is a difference from Cilk and OpenMP task semantics.

    >>> async with finish():
    ...     @spawn()
    ...     def task():
    ...         @spawn()
    ...         def subtask():
    ...              code
    ...         code
    ... # After the finish block, task will be complete, but subtask may not be.

    """
    my_tasks = []
    _task_locals.task_scopes.append(my_tasks)
    try:
        yield
    finally:
        removed_tasks = _task_locals.task_scopes.pop()
        assert removed_tasks is my_tasks
        await tasks(my_tasks)
