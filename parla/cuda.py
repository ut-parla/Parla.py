import threading
from contextlib import contextmanager

import logging
from functools import wraps, lru_cache
from typing import Dict, List, Optional, Tuple, Collection

from parla import array
from parla.array import ArrayType
from . import device
from .device import *
from .environments import EnvironmentComponentInstance, TaskEnvironment, EnvironmentComponentDescriptor

import numpy
import os

try:
    import cupy
    import cupy.cuda
except (ImportError, AttributeError):
    import inspect
    # Ignore the exception if the stack includes the doc generator
    if all("sphinx" not in f.filename for f in inspect.getouterframes(inspect.currentframe())):
        raise
    cupy = None

logger = logging.getLogger(__name__)
profile_flag = bool(os.getenv("PARLA_PROFILE_MEMORY", default=0))

if profile_flag and cupy is not None:
    print("MEMORY PROFILER IS: ACTIVE")
    n_gpus = cupy.cuda.runtime.getDeviceCount()
    mempool = cupy.get_default_memory_pool()
else:
    mempool = None

#Note: Python lists are thread safe! I'm surprised too.
#Extending this list may cause an overhead on runs with a lot of tasks.
#Please don't use this system to profile memory usage on large applications
#Use a tool design for the job like Nvidia Profilers or Remora. This is only for internal development and microbenchmark analysis.

#TODO: Is a class static faster than accessing a global variable? This is arguably cleaner for the namespace though.
class MemPoolLog():
    use_log = list()
    alloc_log = list()


    def append(self, a, b):
        self.use_log.append(a)
        self.alloc_log.append(b)

    def get(self):
        return (self.use_log, self.alloc_log)

    def clean(self):
        self.use_log.clear()
        self.alloc_log.clear()

def log_memory():
    if profile_flag:
        mempool_log = MemPoolLog()
        current_usage = 0
        current_alloc = 0
        for i in range(n_gpus):
            with cupy.cuda.Device(i):
                current_usage += mempool.used_bytes()
                current_alloc += mempool.total_bytes()
        mempool_log.append(current_usage, current_alloc)



__all__ = ["gpu", "GPUComponent", "MultiGPUComponent"]


def _wrap_for_device(ctx: "_GPUDevice", f):
    @wraps(f)
    def ff(*args, **kwds):
        with ctx._device_context():
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

    def asarray_async(self, src):
        if isinstance(src, cupy.ndarray) and src.device.id == self.device.index:
            return src
        if not (src.flags['C_CONTIGUOUS'] or src.flags['F_CONTIGUOUS']):
            raise NotImplementedError('Only contiguous arrays are currently supported for gpu-gpu transfers.')
        dst = cupy.empty_like(src)
        dst.data.copy_from_device_async(src.data, src.nbytes)
        return dst

    def __call__(self, target):
        # TODO Several threads could share the same device object.
        #      It causes data race and CUDA context is incorrectly set.
        #      For now, this remove assumes that one device is always
        #      assigned to one task.
        # FIXME This code breaks the semantics since a different device
        #       could copy data on the current device to a remote device.
        #with self.device._device_context():
        with cupy.cuda.Device(self.device.index):
            if isinstance(target, numpy.ndarray):
                logger.debug("Moving data: CPU => %r", cupy.cuda.Device())
                return cupy.asarray(target)
            elif isinstance(target, cupy.ndarray) and \
                 cupy.cuda.Device() != getattr(target, "device", None):
                logger.debug("Moving data: %r => %r",
                             getattr(target, "device", None), cupy.cuda.Device())
                return self.asarray_async(target)
            else:
                return target


class _GPUDevice(Device):
    @property
    @lru_cache(None)
    def resources(self) -> Dict[str, float]:
        dev = cupy.cuda.Device(self.index)
        free, total = dev.mem_info
        attrs = dev.attributes
        return dict(threads=attrs["MultiProcessorCount"] * attrs["MaxThreadsPerMultiProcessor"], memory=total, vcus=1)

    @property
    def default_components(self) -> Collection["EnvironmentComponentDescriptor"]:
        return [GPUComponent()]

    @contextmanager
    def _device_context(self):
        with self.cupy_device:
            yield

    @property
    def cupy_device(self):
        return cupy.cuda.Device(self.index)

    @lru_cache(None)
    def memory(self, kind: MemoryKind = None):
        return _GPUMemory(self, kind)

    def __repr__(self):
        return "<CUDA {}>".format(self.index)


class _GPUArchitecture(Architecture):
    _devices: List[_GPUDevice]

    def __init__(self, name, id):
        super().__init__(name, id)
        devices = []
        if not cupy:
            self._devices = []
            return
        for device_id in range(cupy.cuda.runtime.getDeviceCount()):
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

    @property
    def num_devices(self):
        return len(self._devices)

    def __call__(self, index, *args, **kwds):
        return _GPUDevice(self, index, *args, **kwds)


gpu = _GPUArchitecture("GPU", "gpu")
gpu.__doc__ = """The `~parla.device.Architecture` for CUDA GPUs.

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

# Integration with parla.environments

class _GPUStacksLocal(threading.local):
    _event_stack: List[cupy.cuda.Event]
    _stream_stack: List[cupy.cuda.Stream]
    _device_stack: List[cupy.cuda.Device]

    def __init__(self):
        super(_GPUStacksLocal, self).__init__()
        self._event_stack = []
        self._stream_stack = []
        self._device_stack = []

    def push_event(self, event):
        self._event_stack.append(event)

    def pop_event(self) -> cupy.cuda.Event:
        return self._event_stack.pop()

    def push_stream(self, stream):
        self._stream_stack.append(stream)

    def pop_stream(self) -> cupy.cuda.Stream:
        return self._stream_stack.pop()

    def push_device(self, dev):
        self._device_stack.append(dev)

    def pop_device(self) -> cupy.cuda.Device:
        return self._device_stack.pop()

    @property
    def stream(self):
        if self._stream_stack:
            return self._stream_stack[-1]
        else:
            return None
    @property
    def device(self):
        if self._device_stack:
            return self._device_stack[-1]
        else:
            return None
    @property
    def event(self):
        if self._event_stack:
            return self._event_stack[-1]
        else:
            return None


class GPUComponentInstance(EnvironmentComponentInstance):
    _object_stack: _GPUStacksLocal
    gpus: List[_GPUDevice]

    def __init__(self, descriptor: "GPUComponent", env: TaskEnvironment):
        super().__init__(descriptor)
        self.gpus = [d for d in env.placement if isinstance(d, _GPUDevice)]
        assert len(self.gpus) == 1
        self.gpu = self.gpus[0]
        # Use a stack per thread per GPU component just in case.
        self._object_stack = _GPUStacksLocal()
        self.event = None

    def _make_stream(self):
        with self.gpu.cupy_device:
            return cupy.cuda.Stream(null=False, non_blocking=True)

    def __enter__(self):
        _gpu_locals._gpus = self.gpus
        # When the context is entered first time,
        # the runtime creates device, stream and event objects.
        # After that, the context reuses the objects until
        # the last context entrace.
        dev = self.gpu.cupy_device
        self._object_stack.push_device(dev)
        dev.__enter__()
        stream = self._make_stream()
        self._object_stack.push_stream(stream)
        stream.__enter__()
        # Create an event.
        # It initialized an event to 'Occurred'.
        # Event recording changes this event to
        # 'Unoccurred' and it is again changed
        # to 'Occurred' when the HEAD of the stream queue
        # points to that record operation.
        event = self.create_event()
        self._object_stack.push_event(event)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dev = self._object_stack.device
        stream = self._object_stack.stream
        event = self._object_stack.event
        try:
            ret = True
            # Exit a stream only if the current
            # context is permanantely exited.
            log_memory()
            stream.__exit__(exc_type, exc_val, exc_tb)
            _gpu_locals._gpus = None
            ret = dev.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._object_stack.pop_event()
            self._object_stack.pop_stream()
            self._object_stack.pop_device()
        return ret

    def get_event_object(self) -> Tuple[str, cupy.cuda.Event]:
        """ Return an event managed by the current stream.
            It is returned as a tuple of architecture type
            and event and therefore, TaskEnvironment requests
            dependee tasks to this task to wait those events on
            the proper devices """
        event = self._object_stack.event
        return ("GPU", event)

    def create_event(self):
        event = cupy.cuda.Event()
        stream = self._object_stack.stream
        return event

    def record_event(self):
        event = self._object_stack.event
        stream = self._object_stack.stream
        event.record(stream)

    def sync_event(self):
        event = self._object_stack.event
        event.synchronize()

    def wait_event(self, event):
        stream = self._object_stack.stream
        stream.wait_event(event)

    def check_device_type(self, arch_type_str):
        """ Check if returned events of `get_event_object()`
            on dependent tasks are the current device type (CUDA).
            To do this, it compares an architecture type strings of that
            events """
        if (arch_type_str == "GPU"):
            return True
        return False

    def initialize_thread(self) -> None:
        for gpu in self.gpus:
            # Trigger cuBLAS/etc. initialization for this GPU in this thread.
            with cupy.cuda.Device(gpu.index) as device:
                a = cupy.asarray([2.])
                cupy.cuda.get_current_stream().synchronize()
                with cupy.cuda.Stream(False, True) as stream:
                    cupy.asnumpy(cupy.sqrt(a))
                    device.cublas_handle
                    device.cusolver_handle
                    device.cusolver_sp_handle
                    device.cusparse_handle
                    stream.synchronize()
                    device.synchronize()

class GPUComponent(EnvironmentComponentDescriptor):
    """A single GPU CUDA component which configures the environment to use the specific GPU using a single
    non-blocking stream

    """

    def combine(self, other):
        assert isinstance(other, GPUComponent)
        return self

    def __call__(self, env: TaskEnvironment) -> GPUComponentInstance:
        return GPUComponentInstance(self, env)


class _GPULocals(threading.local):
    _gpus: Optional[Collection[_GPUDevice]]

    def __init__(self):
        super(_GPULocals, self).__init__()
        self._gpus = None

    @property
    def gpus(self):
        if self._gpus:
            return self._gpus
        else:
            raise RuntimeError("No GPUs configured for this context")

_gpu_locals = _GPULocals()

def get_gpus() -> Collection[Device]:
    return _gpu_locals.gpus


class MultiGPUComponentInstance(EnvironmentComponentInstance):
    gpus: List[_GPUDevice]

    def __init__(self, descriptor: "MultiGPUComponent", env: TaskEnvironment):
        super().__init__(descriptor)
        self.gpus = [d for d in env.placement if isinstance(d, _GPUDevice)]
        assert self.gpus

    def __enter__(self):
        assert _gpu_locals._gpus is None
        _gpu_locals._gpus = self.gpus
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _gpu_locals._gpus == self.gpus
        _gpu_locals._gpus = None
        return False

    def initialize_thread(self) -> None:
        for gpu in self.gpus:
            # Trigger cuBLAS/etc. initialization for this GPU in this thread.
            with cupy.cuda.Device(gpu.index) as device:
                a = cupy.asarray([2.])
                cupy.cuda.get_current_stream().synchronize()
                with cupy.cuda.Stream(False, True) as stream:
                    cupy.asnumpy(cupy.sqrt(a))
                    device.cublas_handle
                    device.cusolver_handle
                    device.cusolver_sp_handle
                    device.cusparse_handle
                    stream.synchronize()
                    device.synchronize()


def get_memory_log():
    mempool_log = MemPoolLog()
    return mempool_log.get()

def clean_memory():
    mempool_log = MemPoolLog()
    mempool_log.clean()

def summarize_memory():
    import numpy as np
    log, alloc = get_memory_log()
    if len(log) > 0:
        print("The max memory usage is: ", np.max(log), " bytes", flush=True)
        print("The max memory alloc is: ", np.max(alloc), "bytes", flush=True)

class MultiGPUComponent(EnvironmentComponentDescriptor):
    """A multi-GPU CUDA component which exposes the GPUs to the task via `get_gpus`.

    The task code is responsible for selecting and using the GPUs and any associated streams.
    """

    def combine(self, other):
        assert isinstance(other, MultiGPUComponent)
        return self

    def __call__(self, env: TaskEnvironment) -> MultiGPUComponentInstance:
        return MultiGPUComponentInstance(self, env)
