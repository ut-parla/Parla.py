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
from abc import abstractmethod, ABCMeta
from contextlib import asynccontextmanager, contextmanager, ExitStack
from typing import Awaitable, Collection, Iterable, Optional, Any, Union, List, FrozenSet, Dict

from parla.device import Device, Architecture, get_all_devices
from parla.task_runtime import TaskID, TaskCompleted, TaskRunning, TaskAwaitTasks, TaskState, DeviceSetRequirements, Task, get_scheduler_context, task_locals
from parla.utils import parse_index
from parla.dataflow import Dataflow

try:
    from parla import task_runtime, array
except ImportError as e:
    # Ignore the exception if the stack includes the doc generator
    if all("sphinx" not in f.filename for f in inspect.getouterframes(inspect.currentframe())):
        raise

logger = logging.getLogger(__name__)

__all__ = [
    "TaskID", "TaskSpace", "spawn", "get_current_devices", "tasks", "finish", "CompletedTaskSpace", "Task", "reserve_persistent_memory"
]


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
        parse_index((), index, lambda x, i: x + (i,),
                lambda x: ret.append(self._data.setdefault(x, TaskID(self._name, x))))
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


# TODO (bozhi): We may need a centralized typing module to reduce types being imported everywhere.
PlacementSource = Union[Architecture, Device, Task, TaskID, Any]

# TODO (bozhi): We may need a `placement` module to hold these `get_placement_for_xxx` interfaces, which makes more sense than the `tasks` module here. Check imports when doing so.
def get_placement_for_value(p: PlacementSource) -> List[Device]:
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


def get_placement_for_set(placement: Collection[PlacementSource]) -> FrozenSet[Device]:
    if not isinstance(placement, Collection):
        raise TypeError(type(placement))
    return frozenset(d for p in placement for d in get_placement_for_value(p))


def get_placement_for_any(placement: Union[Collection[PlacementSource], Any, None]) \
        -> FrozenSet[Device]:
    if placement is not None:
        ps = placement if isinstance(placement, Iterable) and not array.is_array(placement) else [placement]
        return get_placement_for_set(ps)
    else:
        return frozenset(get_all_devices())


def _task_callback(task, body) -> TaskState:
    """
    A function which forwards to a python function in the appropriate device context.
    """
    try:
        body = body

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
          placement: Union[Collection[PlacementSource], Any, None] = None,
          ndevices: int = 1,
          tags: Collection[Any] = (),
          data: Collection[Any] = None,
          input: Collection[Any] = (),
          output: Collection[Any] = (),
          inout: Collection[Any] = ()
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
        taskid = TaskID("global_" + str(len(task_locals.global_tasks)), (len(task_locals.global_tasks),))
        task_locals.global_tasks += [taskid]

    def decorator(body):
        nonlocal placement, memory
        if data is not None:
            if placement is not None or memory is not None:
                raise ValueError("The data parameter cannot be combined with placement or memory paramters.")
            placement = data
            memory = array.storage_size(*data)

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

        # gather input/output/inout, which is hint for data from or to the this task
        dataflow = Dataflow(input, output, inout)

        # Spawn the task via the Parla runtime API
        task = task_runtime.get_scheduler_context().spawn_task(
            function=_task_callback,
            args=(separated_body,),
            deps=deps,
            taskid=taskid,
            req=req,
            dataflow=dataflow,
            name=getattr(body, "__name__", None))

        logger.debug("Created: %s %r", taskid, body)

        for scope in task_locals.task_scopes:
            scope.append(task)

        # Return the task object
        return task

    return decorator


# TODO (bozhi): Why not put it in task_runtime? Remember to update __all__ and clean up imports to do so.
def get_current_devices() -> List[Device]:
    """
    :return: A list of `devices<parla.device.Device>` assigned to the current task. This will have one element unless `ndevices` was \
      provided when the task was `spawned<spawn>`.
    """
    return list(task_runtime.get_devices())


@contextmanager
def _reserve_persistent_memory(memsize, device):
    resource_pool = get_scheduler_context().scheduler._available_resources
    resource_pool.allocate_resources(device, {'memory' : memsize}, blocking = True)
    try:
        yield
    finally:
        resource_pool.deallocate_resources(device, {'memory' : memsize})

# TODO: Move this to parla.device and import it from there. It's generally useful.
def _get_parla_device(device):
    if isinstance(device, Device):
        return device
    try:
        import cupy
    except ImportError:
        pass
    else:
        if isinstance(device, cupy.cuda.Device):
            from .cuda import gpu
            index = device.id
            return gpu(index)
    raise ValueError("Don't know how to convert object of type {} to a parla device object.".format(type(device)))


@contextmanager
def reserve_persistent_memory(amount, device = None):
    """
    :param amount: The number of bytes reserved in the scheduler from tasks for persitent data. \
      This exists, not as any kind of enforced limit on allocation, but rather to let the scheduler \
      have an accurate measure of memory occupancy on the GPU beyond just memory that's used \
      only during a task's execution. It can be specified as an integer representing the nubmer of \
      bytes, an ndarray (cupy or numpy), or a list of ndarrays.
    :param device: The device object where memory is to be reserved. \
      This must be supplied if amount is an integer \
      and may be supplied for an array. In the case of a list or other iterable it must \
      be supplied if any element of the list is not an array. This may be a list of \
      devices if amount is a list of array objects.
    """
    # TODO: This function should be split up into simpler subunits.
    # How exactly that should be done isn't settled yet, but there's
    # some discussion on this at
    # https://github.com/ut-parla/Parla.py/pull/40#discussion_r608857593
    # https://github.com/ut-parla/Parla.py/pull/40#discussion_r608853345
    # TODO: reduce nesting by separating out the try/except idioms for
    # checking if something supports the buffer protocol and checking
    # whether or not something is iterable into separate functions.
    # TODO: Generalize the naming/interface here to allow reserving
    # resources other than memory.
    from . import cpu
    if isinstance(amount, int):
        memsize = amount
    elif hasattr(amount, '__cuda_array_interface__'):
        import cupy
        if not isinstance(amount, cupy.ndarray):
            raise NotImplementedError("Currently only CuPy arrays are supported for making space reservations on the GPU.")
        memsize = amount.nbytes
        if device is None:
            device = amount.device
    else:
        # Check if "amount" supports the buffer protocol.
        # if it does, we're reserving memory on the CPU
        # unless the user says otherwise. If it does not,
        # then assume it's a list of amount parameters
        # that need to be handled individually.
        amount_must_be_iterable = False
        try:
            view = memoryview(amount)
        except TypeError:
            amount_must_be_iterable = True
        else:
            memsize = view.nbytes
            if device is None:
                device = cpu(0)
        # Not a cpu array, so try handling amount as
        # an iterable of things that each need to be processed.
        if amount_must_be_iterable:
            try:
                iter(amount)
            except TypeError as exc:
                raise ValueError("Persistent memory spec is not an integer, array, or iterable object") from exc
            if device is None:
                with ExitStack() as stack:
                    for arr in amount:
                        inner_must_be_iterable = False
                        try:
                            arr.__cuda_array_interface__
                        except AttributeError as exc:
                            inner_must_be_iterable = True
                        else:
                            stack.enter_context(reserve_persistent_memory(arr))
                        if inner_must_be_iterable:
                            try:
                                iter(arr)
                            except TypeError as exc:
                                # TODO: Just use parla.array.get_memory(a).device instead of this manual mess.
                                raise ValueError("Implicit location specification only supported for GPU arrays.") from exc
                            else:
                                stack.enter_context(reserve_persistent_memory(arr))
                    yield
                    return
            device_must_be_iterable = False
            try:
                device = _get_parla_device(device)
            except ValueError:
                device_must_be_iterable = True
            if device_must_be_iterable:
                with ExitStack() as stack:
                    # TODO: do we actually want to support this implicit zip?
                    for arr, dev in zip(amount, device):
                        stack.enter_context(reserve_persistent_memory(arr, dev))
                    yield
                    return
            else:
                with ExitStack() as stack:
                    for arr in amount:
                        stack.enter_context(reserve_persistent_memory(arr, device))
                    yield
                    return
            assert False
    if device is None:
        raise ValueError("Device cannot be inferred.")
    device = _get_parla_device(device)
    if isinstance(device, cpu._CPUDevice):
        raise ValueError("Reserving space for persistent data in main memory is not yet supported.")
    with _reserve_persistent_memory(memsize, device):
        yield


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
    task_locals.task_scopes.append(my_tasks)
    try:
        yield
    finally:
        removed_tasks = task_locals.task_scopes.pop()
        assert removed_tasks is my_tasks
        await tasks(my_tasks)
