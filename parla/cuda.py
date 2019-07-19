from . import device
from .device import *

try:
    import cupy
except ImportError as e:
    import inspect
    # Ignore the exception if the stack includes the doc generator
    if all("sphinx" not in f.filename for f in inspect.getouterframes(inspect.currentframe())):
        raise


class _GPUMemory(Memory):
    def array(self, *args, **kwds):
        return cupy.array(*args, **kwds)

    def __call__(self, target):
        return cupy.asarray(target)


class _GPUDevice(Device):
    def __init__(self, architecture, device_number, **kwds):
        super().__init__(architecture, *(device_number,), **kwds)
        self.device_number = device_number

    @contextmanager
    def context(self):
        with cupy.cuda.Device(self.device_number):
            yield

    def memory(self, kind: MemoryKind = None):
        return _GPUMemory(self, kind)


class _GPUArchitecture(Architecture):
    def __call__(self, *args, **kwds):
        return _GPUDevice(self, *args, **kwds)


gpu = _GPUArchitecture("GPU", "gpu")
gpu.__doc__ = """The `Architecture` for CUDA GPUs.

>>> gpu(0)
"""

device._register_archecture("gpu", gpu)
