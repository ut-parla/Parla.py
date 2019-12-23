import logging
import os
from abc import abstractmethod
from typing import Dict

import numpy

from parla import array
from parla.array import ArrayType
from parla.device import Memory, Device, MemoryKind, Gib

logger = logging.getLogger(__name__)

_MEMORY_FRACTION = 15/16 # The fraction of total memory Parla should assume it can use.


def get_n_cores():
    return len(os.sched_getaffinity(0))


def get_total_memory():
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')


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
        super().__init__(architecture, index, *args, **kws)
        self.n_cores = n_cores or get_n_cores()
        self.available_memory = get_total_memory()*_MEMORY_FRACTION / get_n_cores() * self.n_cores

    @property
    def resources(self) -> Dict[str, float]:
        return dict(threads=self.n_cores, memory=self.available_memory)

    def memory(self, kind: MemoryKind = None):
        return _CPUMemory(self, kind)

    def __repr__(self):
        return "<CPU {} {}>".format(self.index, self.architecture)


class _NumPyArrayType(ArrayType):
    def can_assign_from(self, a, b):
        return isinstance(b, numpy.ndarray)

    def get_memory(self, a):
        return _CPUMemory(None)

    def get_array_module(self, a):
        return numpy


array._register_array_type(numpy.ndarray, _NumPyArrayType())
