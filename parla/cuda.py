from contextlib import contextmanager

import logging
from functools import wraps, lru_cache

from . import device
from .device import *

logger = logging.getLogger(__name__)

try:
    import cupy
except (ImportError, AttributeError):
    import inspect
    # Ignore the exception if the stack includes the doc generator
    if all("sphinx" not in f.filename for f in inspect.getouterframes(inspect.currentframe())):
        raise

__all__ = ["gpu"]


def _wrap_for_device(ctx, f):
    @wraps(f)
    def ff(*args, **kwds):
        with ctx.context():
            return f(*args, **kwds)
    return ff


class _DeviceCUPy:
    def __init__(self, ctx):
        self._ctx = ctx

    def __getattr__(self, item):
        v = getattr(cupy, item)
        if callable(v):
            return _wrap_for_device(self._ctx, v)
        return v

class _GPUMemory(Memory):
    @property
    @lru_cache(None)
    def np(self):
        return _DeviceCUPy(self.device)

    def __call__(self, target):
        with self.device.context():
            if cupy.cuda.Device() != getattr(target, "device", None):
                logger.debug("Moving data: %r => %r", getattr(target, "device", None), cupy.cuda.Device())
                return cupy.asarray(target)
            else:
                return target


class _GPUDevice(Device):
    def __init__(self, architecture, device_number, **kwds):
        self.device_number = device_number
        super().__init__(architecture, device_number+1, *(device_number,), **kwds)

    @contextmanager
    def context(self):
        with cupy.cuda.Device(self.device_number):
            yield

    @lru_cache(None)
    def memory(self, kind: MemoryKind = None):
        return _GPUMemory(self, kind)

    def __repr__(self):
        return "<CUDA {}>".format(self.device_number)


class _GPUArchitecture(Architecture):
    @property
    def devices(self):
        devices = []
        for device_id in range(2**16):
            next_device = cupy.cuda.Device(device_id)
            try:
                next_device.compute_capability
            except cupy.cuda.runtime.CUDARuntimeError:
                break
            devices.append(gpu(device_id))
        return devices

    def __call__(self, *args, **kwds):
        return _GPUDevice(self, *args, **kwds)


gpu = _GPUArchitecture("GPU", "gpu")
gpu.__doc__ = """The `Architecture` for CUDA GPUs.

>>> gpu(0)
"""

device._register_archecture("gpu", gpu)
