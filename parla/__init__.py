"""
Parla is a parallel language for orchestrating high-performance array-based programs.
"Orchestration" refers to controlling lower-level operations from a higher-level language.
In this case, Parla orchestrates low-level array operations and other existing high-performance operations (written in C, C++, or FORTRAN).
"""
from typing import Collection

from parla import task_runtime
from parla.device import get_all_devices
from parla.environments import TaskEnvironment
#from parla import parray

__all__ = ["Parla", "multiload", "TaskEnvironment"]


class Parla:
    environments: Collection[TaskEnvironment]
    _sched: task_runtime.Scheduler

    def __init__(self, environments: Collection[TaskEnvironment]=None, scheduler_class=task_runtime.Scheduler, **kwds):
        assert issubclass(scheduler_class, task_runtime.Scheduler)
        i = 0
        task_envs = []
        for d in get_all_devices():
            print("env_no:", i, " is added to Parla environments")
            task_envs.append(TaskEnvironment(placement=[d], env_no=i))
            i+=1
        self.environments = task_envs
        self.scheduler_class = scheduler_class
        self.kwds = kwds

    def __enter__(self):
        if hasattr(self, "_sched"):
            raise ValueError("Do not use the same Parla object more than once.")
        self._sched = self.scheduler_class(self.environments, **self.kwds)
        return self._sched.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._sched.__exit__(exc_type, exc_val, exc_tb)
        finally:
            del self._sched
