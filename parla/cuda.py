from contextlib import contextmanager

import logging
from functools import wraps, lru_cache
from typing import Dict

import numpy

from parla import array
from parla.array import ArrayType
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
    cupy = None

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
    @property
    @lru_cache(None)
    def resources(self) -> Dict[str, float]:
        dev = cupy.cuda.Device(self.index)
        free, total = dev.mem_info
        attrs = dev.attributes
        return dict(threads=attrs["MultiProcessorCount"] * attrs["MaxThreadsPerMultiProcessor"], memory=total,
                    cvus=attrs["MultiProcessorCount"])

    @contextmanager
    def context(self):
        with cupy.cuda.Device(self.index):
            with cupy.cuda.Stream(null=False, non_blocking=True) as stream:
                yield
                stream.synchronize()

    @lru_cache(None)
    def memory(self, kind: MemoryKind = None):
        return _GPUMemory(self, kind)

    def __repr__(self):
        return "<CUDA {}>".format(self.index)


class _GPUArchitecture(Architecture):
    def __init__(self, name, id):
        super().__init__(name, id)
        devices = []
        if not cupy:
            return
        for device_id in range(2**16):
            cupy_device = cupy.cuda.Device(device_id)
            try:
                cupy_device.compute_capability
            except cupy.cuda.runtime.CUDARuntimeError:
                break
            assert cupy_device.id == device_id
            devices.append(self(cupy_device.id))
        self._devices = devices

    @property
    def devices(self):
        return self._devices

    def __call__(self, index, *args, **kwds):
        return _GPUDevice(self, index, *args, **kwds)


gpu = _GPUArchitecture("GPU", "gpu")
gpu.__doc__ = """The `Architecture` for CUDA GPUs.

>>> gpu(0)
"""

device._register_architecture("gpu", gpu)


class _CuPyArrayType(ArrayType):
    def can_assign_from(self, a, b):
        # TODO: We should be able to do direct copies from numpy to cupy arrays, but it doesn't seem to be working.
        # return isinstance(b, (cupy.ndarray, numpy.ndarray))
        return isinstance(b, cupy.ndarray)

    def get_memory(self, a):
        return gpu(a.device.id).memory()

    def get_array_module(self, a):
        return cupy.get_array_module(a)


if cupy:
    array._register_array_type(cupy.ndarray, _CuPyArrayType())