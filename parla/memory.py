"""
Parla supports an abstract memory model for every device.
"""

from enum import Enum
from collections import namedtuple

from .detail import Detail

class Kind(Enum):
    """
    MemoryKinds specify a kind of memory on a device.
    """
    Fast = "local memory or cache prefetched"
    Slow = "DRAM or similar conventional memory"

class Device:
    pass

class Placement(namedtuple("Placement", ("device", "kind"))):
    """
    Memory locations are specified as a device and a memory type:
    The `Device` specifies the device which has direct (or primary) access to the location;
    The `Kind` specifies what kind of memory should be used.

    The `location` detail can be applied to data references (such as an `~numpy.ndarray`) to force the data to be in the location.
    The data reference can reference memory on a different device (e.i. only the data is moved, the reference is not).
    For instance, a reference on the CPU to data on the GPU.

    In both the value and type cases the result of applying the detail must be used since the detail may not be mutating and may instead return a replacement value or type.

    :allocation: Sometimes (if a the placement must change).
    """
    pass

class location(Detail):
    """
    Specify the location data should be stored.

    .. testsetup::
        import numpy as np
    .. code-block:: python

        location(None, Kind.Fast)(np.array([1, 2, 3])) # In fast memory on any device.
    """
    def __init__(self, placement_or_device, kind=None):
        """
        Create a location detail with either a device and a kind or an existing `Placement`.
        """
        if hasattr(placement, "device") and hasattr(placement, "kind") and other is None:
            self.placement = placement
        else:
            self.placement = Placement(placement_or_device, kind)
