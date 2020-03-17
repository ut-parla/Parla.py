import logging
from typing import Dict

# FIXME: This load of numpy will cause problems if it needs to be multiloaded
# from parla import multiload
# with multiload():
import numpy

import os
import psutil

from . import array, device
from .array import ArrayType
from .device import Architecture, Memory, Device, MemoryKind

__all__ = ["cpu"]

logger = logging.getLogger(__name__)


_MEMORY_FRACTION = 15/16 # The fraction of total memory Parla should assume it can use.


def get_n_cores():
    return psutil.cpu_count(logical=False)


def get_total_memory():
    return psutil.virtual_memory().total


class _CPUMemory(Memory):
    @property
    def np(self):
        return numpy

    def __call__(self, target):
        if getattr(target, "device", None) is not None:
            logger.debug("Moving data: %r => CPU", getattr(target, "device", None))
        return array.asnumpy(target)


class _CPUDevice(Device):
    def __init__(self, architecture: "Architecture", index, *args, n_cores, **kws):
        assert n_cores == 1
        super().__init__(architecture, index, *args, **kws)
        self.n_cores = n_cores or get_n_cores()
        self.available_memory = get_total_memory()*_MEMORY_FRACTION / get_n_cores() * self.n_cores

    @property
    def resources(self) -> Dict[str, float]:
        return dict(threads=self.n_cores, memory=self.available_memory, vcus=self.n_cores)

    def memory(self, kind: MemoryKind = None):
        return _CPUMemory(self, kind)

    def __repr__(self):
        return "<CPU {}>".format(self.index)


class _NumPyArrayType(ArrayType):
    def can_assign_from(self, a, b):
        return isinstance(b, numpy.ndarray)

    def get_memory(self, a):
        # TODO: This is an issue since there is no way to attach allocations of CPU arrays to specific CPU devices.
        return _CPUMemory(cpu(0))

    def get_array_module(self, a):
        return numpy


class _CPUCoresArchitecture(Architecture):
    n_cores: int
    """
    The number of cores for which this process has affinity and are exposed as devices.
    """
    def __init__(self, name, id):
        super().__init__(name, id)
        self.n_cores = get_n_cores()
        self._devices = [self(i) for i in range(self.n_cores)]
        logger.info("CPU 'cores mode' enabled. "
                    "Do not use parallel kernels in this mode (it will cause massive over subscription of the CPU). ")
        logger.info("Parla detected {} cores. Parla cannot currently distinguish threads from core. "
                    "Set CPU affinity to only include one thread on each core to fix this issue.".format(self.n_cores))

    @property
    def devices(self):
        return self._devices

    def __call__(self, id, *args, **kwds) -> _CPUDevice:
        return _CPUDevice(self, id, *args, **kwds, n_cores=1)


cpu = _CPUCoresArchitecture("CPU Cores", "cpu")
cpu.__doc__ = """The `~parla.device.Architecture` for CPUs.

>>> cpu()
"""

device._register_architecture("cpu", cpu)
array._register_array_type(numpy.ndarray, _NumPyArrayType())

# Set OpenMP and MKL to use a single thread for calls
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
