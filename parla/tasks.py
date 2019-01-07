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


class Task(namedtuple("Task", ("_underlying", "name"))):
    pass

class TaskSpace(dict):
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

def spawn(*dependencies):
    """Execute the body of the function as a new task. The task will
    execute in parallel with any following code.

    >>> @spawn(T0)
    ... def T1():
    ...     code

    :param dependencies: any number of dependency arguments which may be tasks or iterables of tasks.

    The declared function is the name of the new task and is not
    callable (since it has already been started).

    """
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


# @contextmanager
# def finish():
#     """
#     Execute the body of the `with` normally and then perform a barrier applying to all tasks created.
#     This block has the similar semantics to the ``sync`` in Cilk.

#     >>> with finish():
#     ...     code

#     """
#     yield
