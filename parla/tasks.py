"""
Parla supports simple task parallelism.

.. testsetup::

    T0 = None
    code = None

"""

from numba import cfunc, jit
import threading
from contextlib import contextmanager
from collections import namedtuple

class TaskID:
    """The identity of a task.

    This combines some ID value with the task object itself. The task
    object is assigned by `spawn`. This can be used in place of the
    task object in most places.

    """
    def __init__(self, id):
        """"""
        self._id = id
        self._task = None

    @property
    def task(self):
        """Get the task object associated with this ID.
        """
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

class TaskSpace:
    """A collection of tasks with IDs.

    A `TaskSpace` can be indexed using any hashable values and any
    number of "dimensions" (indicies). If a dimension is indexed with
    numbers then that dimension can be sliced.

    >>> T = TaskSpace()
    ... for i in range(10):
    ...     @spawn(T[i])(T[0:i-1])
    ...     def t():
    ...         code

    This will produce a series of tasks where each depends on all previous tasks.

    :note: `TaskSpace` does not support assignment to indicies.
    """

    def __init__(self):
        """Create an empty TaskSpace.
        """
        self._data = {}

    def __getitem__(self, index):
        """Get the `TaskID` associated with the provided indicies.
        """
        if not hasattr(index, "__iter__") and not isinstance(index, slice):
            index = (index,)
        ret = []
        def traverse(prefix, index):
            if len(index) > 0:
                i, *rest = index
                if hasattr(i, "__iter__"):
                    for v in i:
                        traverse(prefix + (v,), rest)
                else:
                    traverse(prefix + (i,), rest)
            else:
                ret.append(self._data.setdefault(prefix, TaskID(prefix)))
        traverse((), index)
        return ret


class _TaskLocals(threading.local):
    @property
    def ctx(self):
        return getattr(self, "_ctx", None)
    @ctx.setter
    def ctx(self, v):
        self._ctx = v

_task_locals = _TaskLocals()


def spawn(taskid):
    """spawn(taskid)(*dependencies) -> Task

    Execute the body of the function as a new task. The task may start
    executing immediately, so it may execute in parallel with any
    following code.

    >>> @spawn(T1)(T0) # Create task with ID T1 and dependency on T0
    ... def t():
    ...     code

    :param taskid: the ID of the task in a `TaskSpace` or None if the task does not have an ID.
    :param dependencies: any number of dependency arguments which may be `Tasks<Task>`, `TaskIDs<TaskID>`, or iterables of Tasks or TaskIDs.

    The declared task (`t` above) can be used as a dependency for later tasks (in place of the tasks ID).
    This same value is stored into the task space used in `taskid`.

    :see: :ref:`Blocked Cholesky` Example

    """
    def deps(*dependencies):
        def decorator(body):
            # TODO: Numba jit the body function by default?
            # body = jit("void()")(body)

            # Build the callback to be called directly from the Parla runtime
            @cfunc("void(voidptr, pyobject)")
            def callback(ctx, body):
                old_ctx = _tasks_local.ctx
                _tasks_local.ctx = ctx
                body()
                _tasks_local.ctx = old_ctx

            # Compute the flat dependency set (including unwrapping TaskID objects)
            deps = []
            for ds in dependencies:
                if not hasattr(ds, "__iter__"):
                    ds = (ds,)
                for d in ds:
                    if hasattr(d, "task"):
                        d = d.task
                    assert isinstance(d, Task)
                    deps.append(d)

            # Spawn the task via the Parla runtime API
            if _tasks_local.ctx:
                task = create_task(_tasks_local.ctx, callback, body, deps)
            else:
                # BUG: This function MUST take deps and must return a task
                run_generation_task(callback, body)

            # Store the task object in it's ID object
            taskid.task = task

            # Return the task object
            return task
        return decorator
    return deps


# @contextmanager
# def finish():
#     """
#     Execute the body of the `with` normally and then perform a barrier applying to all tasks created.
#     This block has the similar semantics to the ``sync`` in Cilk.

#     >>> with finish():
#     ...     code

#     """
#     yield
