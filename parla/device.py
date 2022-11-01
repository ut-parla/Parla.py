"""
Parla provides an abstract model of compute devices and memories.
The model is used to describe the placement restrictions for computations and storage.
"""

from contextlib import contextmanager
from enum import Enum
from functools import lru_cache
from typing import Optional, List, Mapping, Dict, Iterable, Collection
from abc import ABCMeta, abstractmethod

import logging

from .detail import Detail

__all__ = [
    "MemoryKind", "Memory", "Device", "Architecture", "get_all_devices", "get_all_architectures",
    "get_architecture",
    "kib", "Mib", "Gib"
]

logger = logging.getLogger(__name__)


kib = 1024
Mib = kib*1024
Gib = Mib*1024


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

    def __repr__(self):
        return "<{} {} {}>".format(type(self).__name__, self.device, self.kind)


class Device(metaclass=ABCMeta):
    """
    An instance of `Device` represents a compute device and its associated memory.
    Every device can directly access its own memory, but may be able to directly or indirectly access other devices memories.
    Depending on the system configuration, potential devices include one CPU core or a whole GPU.

    As devices are logical, the runtime may choose to implement two devices using the same hardware.
    """
    architecture: "Architecture"
    index: Optional[int]

    @lru_cache(maxsize=None)
    def __new__(cls, *args, **kwargs):
        return super(Device, cls).__new__(cls)

    def __init__(self, architecture: "Architecture", index, *args, **kwds):
        """
        Construct a new Device with a specific architecture.
        """
        self.architecture = architecture  # parla.cpu_impl.cpu or parla.cuda.gpu TODO(yineng): more architectures could be added in the future
        self.index = index  # index of gpu
        self.args = args
        self.kwds = kwds

    @property
    @abstractmethod
    def resources(self) -> Dict[str, float]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def default_components(self) -> Collection["EnvironmentComponentDescriptor"]:
        raise NotImplementedError()

    def memory(self, kind: MemoryKind = None):
        return Memory(self, kind)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, type(self)) and \
               self.architecture == o.architecture and \
               self.index == o.index and \
               self.args == o.args and \
               self.kwds == o.kwds

    def __hash__(self):
        return hash(self.architecture) + hash(self.index)*37


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

    def __getitem__(self, ind):
        if isinstance(ind, Iterable):
            return [self(i) for i in ind]
        else:
            return self(ind)

    @property
    @abstractmethod
    def devices(self):
        """
        :return: all `devices<Device>` with this architecture in the system.
        """
        pass

    def __parla_placement__(self):
        return self.devices

    # def memory(self, kind: MemoryKind = None):
    #     return Memory(self, kind)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, type(self)) and \
               self.id == o.id and self.name == o.name

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return type(self).__name__


_architectures = {}
_architectures: Mapping[str, Architecture]

_architectures_list = []
_architectures_list: List[Architecture]


def get_architecture(name):
    try:
        return _architectures[name]
    except KeyError:
        raise ValueError("Non-existent architecture: " + name)


def _register_architecture(name, impl):
    if name in _architectures:
        raise ValueError("Architecture {} is already registered".format(name))
    _architectures[name] = impl
    _architectures_list.append(impl)

@lru_cache(maxsize=1)
def get_all_devices() -> List[Device]:
    """
    :return: A list of all Devices in all Architectures.
    """
    return [d for arch in _architectures_list for d in arch.devices]


def get_all_architectures() -> List[Architecture]:
    """
    :return: A list of all Architectures.
    """
    return list(_architectures_list)

def get_parla_device(device):
    if isinstance(device, Device):
        return device
    try:
        import cupy
    except ImportError:
        pass
    else:
        if isinstance(device, cupy.cuda.Device):
            from .cuda import gpu
            index = device.id
            return gpu(index)
    raise ValueError(
        "Don't know how to convert object of type {} to a parla device object.".format(type(device)))

class ResourcePool:

    _devices: Dict[Device, Dict[str, float]]
    _device_lock: Dict[Device, threading.Condition]

    @staticmethod
    def _initial_resources():
        """
        Initialize a resource counter for all devices. MappedResource tracks all values as a float.
        :return: A dictionary of devices to resource counters (dictionary of resource to amount). Amounts are initialized to zero.
        """
        return {dev: {name: amt for name, amt in dev.resources.items()} for dev in get_all_devices()}


    def __init__(self):

        #Initialize resources types to the device default types
        self._devices = self._initial_resources()

        #Initialize per device lock
        self._device_lock = {dev: threading.Condition(threading.Lock()) for dev in self._devices.keys()}


    def get_all_resources(self):
        """
        Query the current state
        Return a dictionary of all devices to a dictionary of all resources to the amount of that resource on that device.
        """
        return self._devices

    def _get_locks(self, device_list: List[Device], ordered=False):
        """
        Try and back-off scheme to get multi-device locks.

        if device_list is ordered by device id everywhere, then this should perform better

        :param device_list: List of devices to lock
        """

        #TODO: The unordered scheme is a great target to port to Cython/C++ for performance

        if ordered:
            for d in device_list:
                success = self._device_lock[d].acquire()
                assert success
            return success
        else:
            while True:

                acquired_locks = []

                for dev in device_list:
                    success = self._device_lock[dev].acquire(blocking=False)

                    if success:
                        #Add to list of acquired locks
                        acquired_locks.append(dev)

                    if not success:
                        #Free all acquired locks
                        for dev in acquired_locks:
                            self._device_lock[dev].release()
                        
                        #TODO(wlr): Force GIL release here?

                        #Restart attempt to get all locks
                        break
                
                #If all locks acquired, return success
                if len(acquired_locks) == len(device_list):
                    return True

    def get_resources_on_device(self, d: Device):
        """
        Get the current resources on a device.

        :param d: The device to get resources for.
        :return: A dictionary of resource names to amounts.
        """
        return self._devices[d].copy()

    def get_resource_on_device(self, d: Device, resource: str):
        """
        Get the current resources on a device.

        :param d: The device to get resources for.
        :param resource: The resource to get.
        :return: The amount of the resource on the device.
        """
        return self._devices[d][resource]

    def use_resources_on_device(self, d: Device, resources: ResourceDict):
        """
        Update the resource counter for a device. This is used to update the resource counter when a task is assigned to a device.
        :param d: The device to update
        :param resources: The resources to update
        """
        with self._device_lock[d]:
            status = self._use_resource_on_device(d, ResourceDict)

        return status

    def _use_resource_on_device(self, d: Device, resources: ResourceDict):
        """
        Update the resource counter for a device. This is used to update the resource counter when a task is assigned to a device.

        For internal use only. Not thread-safe.

        :param d: The device to update
        :param resources: The resources to update
        """
        for name, amt in resources.items():
            self._devices[d][name] -= amt

        #return success
        return True


    def use_resources(self, device_list: List[Device], resources: List[ResourceDict]):
        """
        Update the resource counter for a list of devices.

        :param device_list: The devices to update
        :param resources: The resources to update
        """

        #Acquire lock on device set
        self._get_locks(device_list)

        #Update resources
        status = self._use_resources(device_list, resources)

        #Release lock on device set
        for d in device_list:
            self._device_lock[d].release()

        #Return success
        return status

    def _use_resources(self, device_list: List[Device], resources: List[ResourceDict]):
        """
        Update the resource counter for a list of devices. This is used to update the resource counter when a task is assigned to a device.

        For internal use only. Not thread-safe.

        :param device_list: The devices to update
        :param resources: The resources to update
        """
        for d, r in zip(device_list, resources):
            self._use_resource_on_device(d, r)

        #return success
        return True


    def release_resource_on_device(self, d: Device, resources: ResourceDict):
        """
        Update the resource counter for a device. This is used to update the resource counter when a task is assigned to a device.
        :param d: The device to update
        :param resources: The resources to update
        """
        with self._device_lock[d]:
            self._remove_resource_on_device_mutex(d, ResourceDict)

        #Return success
        return True

    def _release_resource_on_device(self, d: Device, resources: ResourceDict):
        """
        Update the resource counter for a device. This is used to update the resource counter when a task is assigned to a device.

        For internal use only. Not thread-safe. 

        :param d: The device to update
        :param resources: The resources to update
        """
        for name, amt in resources.items():
            self._devices[d][name] += amt

        #Return success
        return True

    def release_resources(self, device_list: List[Device], resources: List[ResourceDict]):
        """
        Update the resource counter for a list of devices.

        :param device_list: The devices to update
        :param resources: The resources to update
        """

        #Acquire lock on device set
        self._get_locks(device_list)

        #Update resources
        status = self._release_resources(device_list, resources)

        #Release lock on device set
        for d in device_list:
            self._device_lock[d].release()

        #Return success
        return status

    def _release_resources(self, device_list: List[Device], resources: List[ResourceDict]):
        """
        Update the resource counter for a list of devices. This is used to update the resource counter when a task is assigned to a device.

        For internal use only. Not thread-safe.

        :param device_list: The devices to update
        :param resources: The resources to update
        """
        for d, r in zip(device_list, resources):
            self._release_resource_on_device(d, r)

        #return success
        return True


class ManagedResources(ResourcePool):
    """
    Class to track resources reserved on a device. This includes state and resource checking to ensure valid usage. 
    There should be a separate instance of this for both persistent and runtime resources.

    Managed resources is a counter:
    Decreases when a resource is reserved.
    Increases when a resource is freed. (Task End, Data Eviction, & Possibly Work Stealing)
    """

    def _check_resources_on_device(self, d: Device, resources: ResourceDict):
        """
        Check if necessary resouces are currently available on a device.
        Note: This is not atomic and does not reserve resources.

        :param d: The device on which resources exist.
        :param resources: The resources to deallocate.
        """

        for name, amount in resources.items():
            dres = self._devices[d]

            #If amount is greater than what is available on the device, return false
            if amount > dres[name]:
                return False

        #All resources in the set are available
        return True

    def _check_resources(self, device_list: List[Device], resources: List[ResourceDict]):
        """
        Check if necessary resouces are currently available on a device set.
        Note: For internal use. Not thread-safe. 

        :param device_list: The devices on which resources exist.
        :param resources: The resources to deallocate.
        """

        is_available = True

        for d, res in zip(device_list, resources):
            if not self._check_resources_on_device(d, res):
                is_available = False
                break

        return is_available

    def check_resources(self, device_list: List[Device], resources: List[ResourceDict]):
        """
        Check if necessary resouces are currently available on a device set.

        :param device_list: The devices on which resources exist.
        :param resources: The resources to deallocate.
        """

        #Acquire lock on all devices
        self._get_locks(device_list)

        is_available = self._check_resources(device_list, resources)

        #Release lock on all devices
        for d in device_list:
            self._device_lock[d].release()

        return is_available

    def use_resources(self, device_list: List[Device], resources: List[ResourceDict]):
        """
        Use resources on a device set.

        :param device_list: The devices on which resources exist.
        :param resources: The resources to deallocate.
        """

        #Acquire lock on all devices
        self._get_locks(device_list)

        is_available = self._check_resources(device_list, resources)

        if is_available:
            self._use_resources(device_list, resources)
        
        #Release lock on all devices
        for d in device_list:
            self._device_lock[d].release()

        return is_available

    def release_resources(self, device_list: List[Device], resources: List[ResourceDict]):
        """
        Use resources on a device set.

        :param device_list: The devices on which resources exist.
        :param resources: The resources to deallocate.
        """

        #Acquire lock on all devices
        self._get_locks(device_list)

        #TODO(wlr): Possibly add a check if we're not releasing more than we started with?
        self._release_resources(device_list, resources)
        
        #Release lock on all devices
        for d in device_list:
            self._device_lock[d].release()

        return True


class MappedResources(ResourcePool):
    """
    Class to track resources (runtime + persistent) that have been mapped to a device.

    Mapped resources is a counter.
    Increases when the mapping decision is made.
    Mapped resources decrease when resources are freed or mapping decision is changed. (Task End, Data Eviction, Work Stealing) 
    All devices start with 0 mapped resources of each type.
    """

    @staticmethod
    def _initial_resources():
        """
        Initialize a resource counter for all devices. MappedResource tracks all values as a float.
        #TODO(wlr): Maybe make vcus a fraction type here as well?

        :return: A dictionary of devices to resource counters (dictionary of resource to amount). Amounts are initialized to zero.
        """
        return {dev: {name: 0.0 for name, amt in dev.resources.items()} for dev in get_all_devices()}

    def _use_resource(self, d: Device, resources: ResourceDict):
        """
        Update the resource counter for a device. This is used to update the resource counter when a task is assigned to a device.
        :param d: The device to update
        :param resources: The resources to update
        """
        for name, amt in resources.items():
            self._devices[d][name] += amt

    def _release_resource_mutex(self, d: Device, resources: ResourceDict):
        """
        Update the resource counter for a device. This is used to update the resource counter when a task is assigned to a device.
        :param d: The device to update
        :param resources: The resources to update
        """
        for name, amt in resources.items():
            self._devices[d][name] -= amt