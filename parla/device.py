"""
Parla provides an abstract model of compute devices and memories.
The model is used to describe the placement restrictions for computations and storage.
"""

from enum import Enum
from contextlib import contextmanager
from collections import namedtuple
from typing import Optional

from .detail import Detail

class Architecture:
    """
    An Architecture instance represents a range of devices that can be used via the same API and can run the same code.
    For example, an architecture could be "host" (representing the CPUs on the system), or "CUDA" (representing CUDA supporting GPUs).
    """
    def __init__(self, name, id):
        """
        Create a new Architecture with a name and the ID which the runtime will use to identify it.
        """
        self.name = name
        self.id = id

    def __call__(self, *args, **kwds):
        """
        Create a device with this architecture.
        The arguments can specify which physical device you are requesting, but the runtime may override you.

        >>> gpu(0)
        """
        return Device(self, *args, **kwds)

class Device:
    """
    An instance of `Device` represents a **logical** compute device and its associated memory.
    Every device can directly access its own memory, but may be able to directly or indirectly access other devices memories.
    Depending on the system configuration, potential devices include: One NUMA node of a larger system, all CPUs (in multiple NUMA nodes) combined, a whole GPU.

    As devices are logical, the runtime may choose to implement two devices using the same hardware.
    """
    def __init__(self, architecture, *args, **kwds):
        """
        Construct a new **logical** Device with a specific architecture.
        The other arguments can be used to identify specific devices within a system (e.g., a
        """
        self.architecture = architecture
        self.args = args
        self.kwds = kwds

    @contextmanager
    def context(self):
        yield

class MemoryKind(Enum):
    """
    MemoryKinds specify a kind of memory on a device.
    """
    Fast = "local memory or cache prefetched"
    Slow = "DRAM or similar conventional memory"

class Memory(Detail):
    """
    Memory locations are specified as a device and a memory type:
    The `Device` specifies the device which has direct (or primary) access to the location;
    The `Kind` specifies what kind of memory should be used.

    A `Memory` instance can also be used as a detail on data references (such as an `~numpy.ndarray`) to force the data to be in the location.

    .. testsetup::
        import numpy as np
    .. code-block:: python

        Memory(gpu, MemoryKind.Fast)(np.array([1, 2, 3])) # In fast memory on a GPU.
        Memory(gpu(0), MemoryKind.Slow)(np.array([1, 2, 3])) # In slow memory on GPU #0.

    :allocation: Sometimes (if a the placement must change).
    """
    def __init__(self, device=None, kind: Optional[MemoryKind] = None):
        """
        :param device: The device which owns this memory (or None meaning any device).
        :type device: A `Device`, `Architecture`, or None.
        :param kind: The kind of memory (or None meaning any kind).
        :type kind: A `MemoryKind` or None.
        """
        self.device = device
        self.kind = kind

cpu = Architecture("CPU", "cpu")

_architectures = {
    "cpu": cpu,
    }

def _get_architecture(name):
    return _architectures[name]

def _register_archecture(name, impl):
    assert name not in _architectures
    _architectures[name] = impl
