from . import device
from .device import *

try:
    import cupy
except ImportError as e:
    import inspect
    # Ignore the exception if the stack includes the doc generator
    if all("sphinx" not in f.filename for f in inspect.getouterframes(inspect.currentframe())):
        raise

class _GPUDevice(Device):
    @contextmanager
    def context(self):
        with cupy.cuda.Device(0):
            yield

class _GPUArchitecture(Architecture):
    def __call__(self, *args, **kwds):
        return _GPUDevice(self, *args, **kwds)

gpu = _GPUArchitecture("GPU", "gpu")
gpu.__doc__ = """The `Architecture` for CUDA GPUs.

>>> gpu(0)
"""

device._register_archecture("gpu", gpu)
