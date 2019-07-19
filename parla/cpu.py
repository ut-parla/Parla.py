import numpy

from . import array
from .device import Memory, Device, MemoryKind, Architecture, _register_archecture


class _CPUMemory(Memory):
    def array(self, *args, **kwds):
        return numpy.array(*args, **kwds)

    def __call__(self, target):
        return array.asnumpy(target)


class _CPUDevice(Device):
    def memory(self, kind: MemoryKind = None):
        return _CPUMemory(self, kind)


class _CPUArchitecture(Architecture):
    def __call__(self, *args, **kwds):
        return _CPUDevice(self, *args, **kwds)


cpu = _CPUArchitecture("CPU", "cpu")
cpu.__doc__ = """The `Architecture` for CPUs.

>>> cpu()
"""

_register_archecture("cpu", cpu)
