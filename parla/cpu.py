from parla._cpuutils import _CPUDevice
from .device import Architecture, _register_architecture

__all__ = ["cpu"]

class _CPUArchitecture(Architecture):

    def __init__(self, name, id):
        super().__init__(name, id)
        self._device = self()

    @property
    def devices(self):
        return [self._device]

    def __call__(self, id=0, *args, **kwds):
        if id != 0:
            raise ValueError("Parla only supports a single CPU device in non-'cores' mode.")
        return _CPUDevice(self, id, *args, **kwds)


cpu = _CPUArchitecture("CPU", "cpu")
cpu.__doc__ = """The `Architecture` for CPUs.

>>> cpu()
"""

_register_architecture("cpu", cpu)
