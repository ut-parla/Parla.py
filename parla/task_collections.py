from typing import Awaitable, Collection, Iterable, Optional, Any, Union, List, FrozenSet, Dict
from abc import abstractmethod, ABCMeta
from contextlib import asynccontextmanager

from parla.utils import parse_index
from parla.task_runtime import TaskID, Task, TaskAwaitTasks, task_locals

__all__ = ["TaskSpace", "CompletedTaskSpace", "finish"]

class TaskSet(Awaitable, Collection, metaclass=ABCMeta):
    """
    A collection of tasks.
    """

    @property
    @abstractmethod
    def _tasks(self) -> Collection:
        pass

    @property
    def _flat_tasks(self) -> List[Union[TaskID, Task]]:
        # Compute the flat dependency set (including unwrapping TaskID objects)
        dependencies = []
        for ds in self._tasks:
            if not isinstance(ds, Iterable):
                ds = (ds,)
            for d in ds:
                if hasattr(d, "task"):
                    if d.task is not None:
                        d = d.task
                # if not isinstance(d, task_runtime.Task):
                #    raise TypeError("Dependencies must be TaskIDs or Tasks: " + str(d))
                dependencies.append(d)
        return dependencies

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