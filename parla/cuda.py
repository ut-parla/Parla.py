from . import device
from .device import *

import cupy

class _GPUDevice(Device):
    @contextmanager
    def context(self):
        with cupy.cuda.Device(0):
            yield

class _GPUArchitecture(Architecture):
    def __call__(self, *args, **kwds):
        return _GPUDevice(self, *args, **kwds)

gpu = _GPUArchitecture("GPU", "gpu")

device._register_archecture("gpu", gpu)
