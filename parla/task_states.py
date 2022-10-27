from abc import abstractmethod, ABCMeta
from typing import Optional, List

class TaskState(object, metaclass=ABCMeta):
    __slots__ = []

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError()


class TaskWaiting(TaskState):
    """ This state specifies that a task is waiting for dependencies' spawnings
    """
    @property
    def is_terminal(self):
        return False


class TaskRunning(TaskState):
    __slots__ = ["func", "args", "dependencies"]

    @property
    def is_terminal(self):
        return False

    # The argument dependencies intentially has no type hint.
    # Callers can pass None if they want to pass empty dependencies.
    def __init__(self, func, args, dependencies: Optional[List]):
        if dependencies is not None:
            # d could be one of four types: Task, DataMovementTask, TaskID or other types.
            #assert all(isinstance(d, (Task, TaskID)) for d in dependencies)
            #self.dependencies = [
            #    d for d in dependencies if isinstance(d, Task)]

            #COMMENT(wlr): I think we shouldn't filter out the TaskID here. Otherwise, we cannot barrier on unspawned tasks
            self.dependencies = dependencies
        else:
            self.dependencies = []
        self.args = args
        self.func = func

    def clear_dependencies(self):
        self.dependencies = []

    def __repr__(self):
        if self.func:
            # return "TaskRunning({}, {}, {})".format(self.func.__name__, self.args, self.dependencies)
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