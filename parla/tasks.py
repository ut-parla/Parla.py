"""
Parla supports simple task parallelism.

.. testsetup::

    T0 = None
    code = None
    from .cpu import cpu

"""

import ctypes
import logging
import threading
from collections import Sized
from collections.abc import Iterable

from parla import device
from parla.device import Device

try:
    from parla import task_runtime
except ImportError as e:
    import inspect
    # Ignore the exception if the stack includes the doc generator
    if all("sphinx" not in f.filename for f in inspect.getouterframes(inspect.currentframe())):
        raise

logger = logging.getLogger(__name__)

__all__ = [
    "TaskID", "TaskSpace", "spawn", "get_current_device"
]


class TaskID:
    """The identity of a task.

    This combines some ID value with the task object itself. The task
    object is assigned by `spawn`. This can be used in place of the
    task object in most places.

    """

    def __init__(self, name, id):
        """"""
        self._name = name
        self._id = id
        self._task = None

    @property
    def task(self):
        """Get the task object associated with this ID.
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

    def __str__(self):
        return "TaskID({}{}, task={})".format(self.name, self.id, self._task)


class TaskSpace:
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

    def __init__(self, name=""):
        """Create an empty TaskSpace.
        """
        self._name = name
        self._data = {}

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


class _TaskLocals(threading.local):
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


class _TaskData:
    def __init__(self, _task_locals, body, dependencies):
        self._task_locals = _task_locals
        self.body = body
        self.dependencies = dependencies

    def __repr__(self):
        return "<TaskData {body}>".format(**self.__dict__)


_task_locals = _TaskLocals()

# _task_callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.py_object)


# @_task_callback_type
def _task_callback(data):
    """
    A function which forwards to a python function in the appropriate device context.
    """
    logger.debug("Starting: %s", data.taskid)
    try:
        with get_current_device().context():
            data.body()
    except:
        print("exiting because of unhandled exception.")
        print("Traceback was:")
        import traceback
        print(traceback.format_exc())
        import sys
        sys.exit()
    finally:
        logger.debug("Finished: %s", data.taskid)
    return 0


def _make_cell(val):
    """
    Create a new Python closure cell object.

    You should not be using this.
    """
    x = val

    def closure():
        return x

    return closure.__closure__[0]


def spawn(taskid: TaskID = None, dependencies=(), *, placement: Device = None):
    """
    Execute the body of the function as a new task. The task may start
    executing immediately, so it may execute in parallel with any
    following code.

    >>> @spawn(T1, [T0]) # Create task with ID T1 and dependency on T0
    ... def t():
    ...     code

    >>> @spawn(T1, [T0], placement=cpu())
    ... def t():
    ...     code

    :param taskid: the ID of the task in a `TaskSpace` or None if the task does not have an ID.
    :param dependencies: any number of dependency arguments which may be `Tasks<Task>`, `TaskIDs<TaskID>`, or iterables of Tasks or TaskIDs.
    :param placement: a device on which the task should run.

    The declared task (`t` above) can be used as a dependency for later tasks (in place of the tasks ID).
    This same value is stored into the task space used in `taskid`.

    :see: :ref:`Blocked Cholesky` Example

    .. todo:: Provide `placement` to parla_task and implement it in the runtime

    """

    def decorator(body):
        nonlocal taskid

        # Compute the flat dependency set (including unwrapping TaskID objects)
        deps = []
        for ds in dependencies:
            if not isinstance(ds, Iterable):
                ds = (ds,)
            for d in ds:
                if hasattr(d, "task"):
                    d = d.task
                assert isinstance(d, task_runtime.Task)
                deps.append(d)

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

        data = _TaskData(_task_locals, separated_body, dependencies)

        if not taskid:
            taskid = TaskID("global", len(_task_locals.global_tasks))
            _task_locals.global_tasks += [taskid]
        taskid.data = data
        taskid.dependencies = dependencies
        data.taskid = taskid

        queue_index = None if placement is None else placement.index

        # Spawn the task via the Parla runtime API
        task = task_runtime.run_task(_task_callback, (data,), deps, queue_identifier=queue_index)

        # Store the task object in it's ID object
        taskid.task = task

        logger.debug("Created: %s <%s, %s, %r>", taskid, placement, queue_index, body)

        # Return the task object
        return task

    return decorator


def get_current_device():
    index = task_runtime.get_device_id()
    type = task_runtime.get_device_type(index)
    if type == "cpu":
        arch = device._get_architecture("cpu")
        arch_index = index
    elif type == "gpu":
        arch = device._get_architecture("gpu")
        arch_index = index - 1
    else:
        raise ValueError("Could not find device for this thread.")
    d = arch(arch_index)
    d.index = index
    return d

# @contextmanager
# def finish():
#     """
#     Execute the body of the `with` normally and then perform a barrier applying to all tasks created.
#     This block has the similar semantics to the ``sync`` in Cilk.

#     >>> with finish():
#     ...     code

#     """
#     yield
