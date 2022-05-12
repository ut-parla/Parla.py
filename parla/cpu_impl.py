import logging
from typing import Dict, Collection

import os
import psutil
from typing import List

from . import array, device
from .array import ArrayType
from .device import Architecture, Memory, Device, MemoryKind
from .environments import EnvironmentComponentInstance, TaskEnvironment, EnvironmentComponentDescriptor

__all__ = ["cpu"]

logger = logging.getLogger(__name__)


# The fraction of total memory Parla should assume it can use.
_MEMORY_FRACTION = 15/16


def get_n_cores():

    cores = os.environ.get("PARLA_CORES", psutil.cpu_count(logical=False))
    return int(cores)


def get_total_memory():
    return psutil.virtual_memory().total


class _CPUMemory(Memory):
    @property
    def np(self):
        return numpy

    def __call__(self, target):
        if getattr(target, "device", None) is not None:
            logger.debug("Moving data: %r => CPU",
                         getattr(target, "device", None))
        return array.asnumpy(target)


class _CPUDevice(Device):
    def __init__(self, architecture: "Architecture", index, *args, n_cores, **kws):
        super().__init__(architecture, index, *args, **kws)
        self.n_cores = n_cores or get_n_cores()
        self.available_memory = get_total_memory()*_MEMORY_FRACTION / \
            get_n_cores() * self.n_cores

    @property
    def resources(self) -> Dict[str, float]:
        return dict(cores=self.n_cores, memory=self.available_memory, vcus=1)

    @property
    def default_components(self) -> Collection["EnvironmentComponentDescriptor"]:
        return [UnboundCPUComponent()]

    def memory(self, kind: MemoryKind = None):
        return _CPUMemory(self, kind)

    def __repr__(self):
        return "<CPU {}>".format(self.index)


class _GenericCPUArchitecture(Architecture):
    def __init__(self, name, id):
        super().__init__(name, id)
        self.n_cores = get_n_cores()


class _CPUCoresArchitecture(_GenericCPUArchitecture):
    """
    A CPU architecture that treats each CPU core as a Parla device.
    Each device will have one VCU.

    WARNING: This architecture configures OpenMP and MKL to execute without any parallelism.
    """

    n_cores: int
    """
    The number of cores for which this process has affinity and are exposed as devices.
    """

    def __init__(self, name, id):
        super().__init__(name, id)
        self._devices = [self(i) for i in range(self.n_cores)]
        logger.warning("CPU 'cores mode' enabled. "
                       "Do not use parallel kernels in this mode (it will cause massive over subscription of the CPU). "
                       "Setting OMP_NUM_THREADS=1 and MKL_THREADING_LAYER=SEQUENTIAL to avoid implicit parallelism.")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

    @property
    def devices(self):
        return self._devices

    def __call__(self, id, *args, **kwds) -> _CPUDevice:
        return _CPUDevice(self, id, *args, **kwds, n_cores=1)


class _CPUWholeArchitecture(_GenericCPUArchitecture):
    """
    A CPU architecture that treats the entire CPU as a single Parla device.
    That device will have one VCU per core.
    """

    n_cores: int
    """
    The number of cores for which this process has affinity and are exposed as VCUs.
    """

    def __init__(self, name, id):
        super().__init__(name, id)
        self._device = self(0)

    @property
    def devices(self):
        return [self._device]

    def __call__(self, id, *args, **kwds) -> _CPUDevice:
        assert id == 0, "Whole CPU architecture only supports a single CPU device."
        return _CPUDevice(self, id, *args, **kwds, n_cores=None)


class UnboundCPUComponentInstance(EnvironmentComponentInstance):
    def __init__(self, descriptor, env):
        super().__init__(descriptor)
        cpus = [d for d in env.placement if isinstance(d, _CPUDevice)]
        assert len(cpus) == 1
        self.cpus = cpus

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def initialize_thread(self) -> None:
        pass

    def get_event_object(self):
        return None

    def create_event(self):
        pass

    def record_event(self):
        pass

    def sync_event(self):
        pass

    def wait_event(self):
        pass

    def wait_event(self, event):
        pass

    def check_device_type(self, checking_type_str):
        if (checking_type_str == "CPU"):
            return True
        return False


class UnboundCPUComponent(EnvironmentComponentDescriptor):
    """A single CPU component that represents a "core" but isn't automatically bound to the given core.
    """

    def combine(self, other):
        assert isinstance(other, UnboundCPUComponent)
        assert self.cpus == other.cpus
        return self

    def __call__(self, env: TaskEnvironment) -> UnboundCPUComponentInstance:
        return UnboundCPUComponentInstance(self, env)


if True or os.environ.get("PARLA_CPU_ARCHITECTURE", "").lower() == "cores":
    cpu = _CPUCoresArchitecture("CPU Cores", "cpu")
else:
    if os.environ.get("PARLA_CPU_ARCHITECTURE", "").lower() not in ("whole", ""):
        logger.warning("PARLA_CPU_ARCHITECTURE only supports cores or whole.")
    cpu = _CPUWholeArchitecture("Whole CPU", "cpu")
cpu.__doc__ = """The `~parla.device.Architecture` for CPUs.

>>> cpu()
"""
