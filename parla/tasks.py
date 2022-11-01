"""
Parla supports simple task parallelism.

.. testsetup::

    T0 = None
    code = None
    from .cpu import cpu

"""
import logging
import inspect
from contextlib import contextmanager, ExitStack
from typing import Collection, Optional, Any, Union

from parla.task_runtime import ComputeTask, TaskID, TaskCompleted, TaskRunning, TaskAwaitTasks, TaskState, DeviceSetRequirements, Task, get_scheduler_context, task_locals, WorkerThread
from parla.dataflow import Dataflow
from parla.placement import PlacementSource, get_placement_for_any
from parla.device import get_parla_device
from parla.task_collections import tasks as Tasks 

try:
    from parla import task_runtime, array
except ImportError as e:
    # Ignore the exception if the stack includes the doc generator
    if all("sphinx" not in f.filename for f in inspect.getouterframes(inspect.currentframe())):
        raise

logger = logging.getLogger(__name__)

__all__ = [
    "TaskID", "TaskSpace", "spawn", "tasks", "finish", "CompletedTaskSpace", "Task", "reserve_persistent_memory"
]


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

                # This body returns information of the function body, not task.
                # It means that any task shares the same function body can use the
                # same information. This should be handled in more detailed.
                # For example, dependencies could include TaskID, not Task objects
                # if threads try to spawn several tasks sharing the same body with
                # different task space, and if one tries to make dependency with
                # nested tasks of the previous task.
                # This could happen since the nested tasks could be spawned in parallel.
                # To be specific, let's consider the below case.
                #
                # outer_task = TaskSpace("outer")
                # inner_task = TaskSpace("inner")
                # for rep in range(reps):
                #     dependency = [inner_task[rep - 1]] if rep > 0 else []
                #     @spawn(out_task[rep], dependencies=[dep])
                #     def t:
                #         @spawn(inner_task[rep]):
                #         def inner_t:
                #         ...
                #
                # In the above case, the main thread will try to spawn
                # out_task[0], and out_task[1] immediately.
                # (Note that there is no await statement in the above)
                #
                # Let's assume that threads try to spawn tasks in this order
                # at the first round:
                # by main thread             -> by another thread
                # out_task[0] -> out_task[1] -> inner_task[0]
                #
                # out_task[1] needs inner_task[0].
                # But inner_task[0] is not yet spawned.
                # out_task[1] is waiting for inner_task[0]'s spawn.
                # out_task[0] gets a thread who runs it.
                # When the thread tries to run out_task[0], it gets meta information
                # from body.
                # Since out_task[1] already requests inner_task[0] as a dependent task,
                # inner_task[0] exists on TaskAwaitTasks, as TaskID, not Task.
                # (Note that Task is mapped to TaskID after the task is spawnd).
                #
                # In this case, Parla removes that unspawned task id from
                # the current dependency list. (in this case, out_task[0])
                # This does not cause a problem because,
                #
                # First, any task whose have unspawned dependencies will never get
                # this place. Also this unspawned dependencies are removed and do not
                # exist on the task's dependency list.
                #
                # Second, if any task is ready to run after all dependencies are spawned,
                # it will re-adds those removed/late spawned dependencies to its dependency
                # list. (which means that removing the dependency from body's
                # dependency list is fine)
                #
                # Therefore, non-Task or DataMovementTask elements of dependencies
                # are fine even though it removes them at HERE.
                #
                # TODO(lhc): if you think wrong, please let me know.
                new_task_info = body.send(in_value)
                task.value_task = None
                if not isinstance(new_task_info, TaskAwaitTasks):
                    raise TypeError(
                        "Parla coroutine tasks must yield a TaskAwaitTasks")
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


def spawn(taskid: Optional[TaskID] = None,
          dependencies=(), *,
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

    :not see: :ref:`Fox's Algorithm` Example

    """
    # :param vcus: The amount of compute power this task uses. It is specified in "Virtual Compute Units".
    # TODO: Document tags argument

    if not taskid:
        taskid = TaskID("global_" + str(len(task_locals.global_tasks)),
                        (len(task_locals.global_tasks),))
        task_locals.global_tasks += [taskid]

    def decorator(body) -> ComputeTask:
        nonlocal placement, memory
        if data is not None:
            if placement is not None or memory is not None:
                raise ValueError(
                    "The data parameter cannot be combined with placement or memory paramters.")
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
            raise TypeError(
                "Spawned tasks must be normal functions or coroutines; not generators.")

        if inspect.iscoroutine(body):
            # An already running coroutine does not need changes since we assume
            # it was changed correctly when the original function was spawned.

            # This is for tasks that are "relaunched"
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

        # Compute the flat dependency set (including unwrapping TaskID objects)
        taskid.dependencies = dependencies

        processed_dependencies = Tasks(*dependencies)._flat_tasks

        # gather input/output/inout, which is hint for data from or to the this task
        # TODO (ses): I gathered these into lists so I could perform concatentation later. This may be inefficient.
        dataflow = Dataflow(list(input), list(output), list(inout))

        # Get handle to current scheduler
        scheduler = task_runtime.get_scheduler_context()

        if isinstance(scheduler, WorkerThread):
            # If we are in a worker thread, get the real scheduler object instead.
            scheduler = scheduler.scheduler

        # Spawn the task via the Parla runtime API
        task = scheduler.spawn_task(
            function=_task_callback,
            args=(separated_body,),
            dependencies=processed_dependencies,
            taskid=taskid,
            req=req,
            dataflow=dataflow,
            name=getattr(body, "__name__", None))

        logger.debug("Created: %s %r", taskid, body)

        #print("task scopes", task_locals.task_scopes, flush=True)
        #for scope in task_locals.task_scopes:
        #    scope.append(task)

        # Activate scheduler
        scheduler.start_scheduler_callbacks()

        # Return the task object to user code
        return task

    return decorator


#TODO(wlr): All of this memory handling should not be in this file. (Also needs to be cleaned up and documented.)
#TODO(wlr): Cannot be moved until get_scheduler_context() can be imported without the full runtime. Python has the worst import system. 

@contextmanager
def _reserve_persistent_memory(memsize, device):
    resource_pool = get_scheduler_context().scheduler._available_resources
    resource_pool.allocate_resources(
        device, {'memory': memsize}, blocking=True)
    try:
        yield
    finally:
        resource_pool.deallocate_resources(device, {'memory': memsize})



@contextmanager
def reserve_persistent_memory(amount, device=None):
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
            raise NotImplementedError(
                "Currently only CuPy arrays are supported for making space reservations on the GPU.")
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
                raise ValueError(
                    "Persistent memory spec is not an integer, array, or iterable object") from exc
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
                                raise ValueError(
                                    "Implicit location specification only supported for GPU arrays.") from exc
                            else:
                                stack.enter_context(
                                    reserve_persistent_memory(arr))
                    yield
                    return
            device_must_be_iterable = False
            try:
                device = get_parla_device(device)
            except ValueError:
                device_must_be_iterable = True
            if device_must_be_iterable:
                with ExitStack() as stack:
                    # TODO: do we actually want to support this implicit zip?
                    for arr, dev in zip(amount, device):
                        stack.enter_context(
                            reserve_persistent_memory(arr, dev))
                    yield
                    return
            else:
                with ExitStack() as stack:
                    for arr in amount:
                        stack.enter_context(
                            reserve_persistent_memory(arr, device))
                    yield
                    return
            assert False
    if device is None:
        raise ValueError("Device cannot be inferred.")
    device = get_parla_device(device)
    if isinstance(device, cpu._CPUDevice):
        raise ValueError(
            "Reserving space for persistent data in main memory is not yet supported.")
    with _reserve_persistent_memory(memsize, device):
        yield

