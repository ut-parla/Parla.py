"""
Parla is a parallel language for orchestrating high-performance array-based programs.
"Orchestration" refers to controlling lower-level operations from a higher-level language.
In this case, Parla orchestrates low-level array operations and other existing high-performance operations (written in C, C++, or FORTRAN).
"""
from typing import Optional

from parla import task_runtime
from parla.device import get_all_devices
from parla.task_runtime import Scheduler


class Parla:
    n_devices: Optional[int]
    n_tasks_per_device: int
    _sched: Scheduler

    def __init__(self, n_devices: Optional[int] = None, n_tasks_per_device: int = 2, **kwds):
        self.n_tasks_per_device = n_tasks_per_device
        self.n_devices = n_devices
        self.kwds = kwds

    def __enter__(self):
        n_devices = self.n_devices or len(get_all_devices())
        self._sched = task_runtime.Scheduler(n_threads=n_devices * self.n_tasks_per_device, **self.kwds)
        return self._sched.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._sched.__exit__(exc_type, exc_val, exc_tb)
        finally:
            del self._sched