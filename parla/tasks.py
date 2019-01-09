"""
Parla supports simple task parallelism.

.. testsetup::

    T0 = None
    code = None

"""

import ctypes
import threading
from contextlib import contextmanager
from collections import namedtuple


class Task(namedtuple("Task", ("underlying", "name"))):
    pass

class TaskSpace(dict):
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

    """
    def __getitem__(self, index):
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
                ret.append(self[prefix])
        traverse((), index)
        return ret



_task_callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_tasks_local = threading.local()

def spawn(taskid):
    """@spawn(taskid)(\*dependencies)

    Execute the body of the function as a new task. The task may start
    executing immediately, so it may execute in parallel with any
    following code.

    >>> @spawn(T1)(T0) # Create task with ID T1 and dependency on T0
    ... def t():
    ...     code

    :param taskid: the ID of the task in a `TaskSpace` or None if the task does not have an ID.
    :param dependencies: any number of dependency arguments which may be tasks, ids, or iterables of tasks or ids.

    The declared task (`t` above) can be used as a dependency for later tasks (in place of the tasks ID).
    This same value is conceptually stored into the task space used in `taskid`.

    :see: `Blocked Cholesky Example <https://github.com/UTexas-PSAAP/Parla.py/blob/master/examples/blocked_cholesky.py#L37>`_

    """
    def deps(*dependencies):
        def decorator(body):
            @_task_callback_type
            def callback(task):
                _tasks_local.new_tasks = []
                body()
                for t in _tasks_local.new_tasks:
                    parla_task_add_child(task, t)
                _tasks_local.new_tasks = None
                return 0
            task = parla_new_task(callback)
            for ds in dependencies:
                if not hasattr(ds, "__iter__"):
                    ds = (ds,)
                for d in ds:
                    assert isinstance(d, Task)
                    parla_task_add_dependency(task, d._underlying)
            if _tasks_local.new_tasks is not None:
                _tasks_local.new_tasks.append(task)
            else:
                # We are top level so spawn this task directly
                parla_task_ready(task)
            return Task(task, body.__name__)
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
