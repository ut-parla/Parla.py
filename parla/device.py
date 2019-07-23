"""
Parla provides an abstract model of compute devices and memories.
The model is used to describe the placement restrictions for computations and storage.
"""

from contextlib import contextmanager
from enum import Enum
from functools import lru_cache
from typing import Optional
from abc import ABCMeta, abstractmethod, abstractproperty

import logging

from .detail import Detail

logger = logging.getLogger(__name__)

class MemoryKind(Enum):
    """
    MemoryKinds specify a kind of memory on a device.
    """
    Fast = "local memory or cache prefetched"
    Slow = "DRAM or similar conventional memory"


class Memory(Detail, metaclass=ABCMeta):
    """
    Memory locations are specified as a device and a memory type:
    The `Device` specifies the device which has direct (or primary) access to the location;
    The `Kind` specifies what kind of memory should be used.

    A `Memory` instance can also be used as a detail on data references (such as an `~numpy.ndarray`) to copy the data
    to the location. If the original object is already in the correct location, it is returned unchanged, otherwise
    a copy is made in the correct location.
    There is no relationship between the original object and the new one, and the programmer must copy data back to the
    original if needed.

    .. testsetup::
        import numpy as np
    .. code-block:: python

        gpu.memory(MemoryKind.Fast)(np.array([1, 2, 3])) # In fast memory on a GPU.
        gpu(0).memory(MemoryKind.Slow)(np.array([1, 2, 3])) # In slow memory on GPU #0.

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

    @property
    @abstractmethod
    def np(self):
        """
        Return an object with an interface similar to the `numpy` module, but
        which operates on arrays in this memory.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, target):
        """
        Copy target into this memory.

        :param target: A data object (e.g., an array).
        :return: The copied data object in this memory. The returned object should have the same interface as the original.
        """
        raise NotImplementedError()


class Device(metaclass=ABCMeta):
    """
    An instance of `Device` represents a **logical** compute device and its associated memory.
    Every device can directly access its own memory, but may be able to directly or indirectly access other devices memories.
    Depending on the system configuration, potential devices include: One NUMA node of a larger system, all CPUs (in multiple NUMA nodes) combined, a whole GPU.

    As devices are logical, the runtime may choose to implement two devices using the same hardware.
    """
    index: Optional[int]

    @lru_cache(maxsize=None)
    def __new__(cls, *args, **kwargs):
        return super(Device, cls).__new__(cls)

    def __init__(self, architecture, index, *args, **kwds):
        """
        Construct a new **logical** Device with a specific architecture.
        """
        self.architecture = architecture
        self.index = index
        self.args = args
        self.kwds = kwds

    @contextmanager
    def context(self):
        yield

    def memory(self, kind: MemoryKind = None):
        return Memory(self, kind)


class Architecture(metaclass=ABCMeta):
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

    # def memory(self, kind: MemoryKind = None):
    #     return Memory(self, kind)


_architectures = {}


def _get_architecture(name):
    return _architectures[name]


def _register_archecture(name, impl):
    assert name not in _architectures
    _architectures[name] = impl
