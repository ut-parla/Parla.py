import logging
from abc import ABCMeta, abstractmethod
from typing import Dict
import copy as cp
# FIXME: This load of numpy will cause problems if it needs to be multiloaded
# from parla import multiload
# with multiload():
import numpy as np

from parla.device import Memory
from parla.tasks import get_current_devices

logger = logging.getLogger(__name__)

__all__ = ["get_array_module", "get_memory", "is_array", "asnumpy", "copy", "clone_here", "storage_size","get_current_device_copy"]



class ArrayType(metaclass=ABCMeta):
    @abstractmethod
    def get_memory(self, a):
        """
        :param a: An array of self's type.
        :return: The memory containing `a`.
        """
        pass

    @abstractmethod
    def can_assign_from(self, a, b):
        """
        :param a: An array of self's type.
        :param b: An array of any type.
        :return: True iff `a` supports assignments from `b`.
        """
        pass

    @abstractmethod
    def get_array_module(self, a):
        """
        :param a: An array of self's type.
        :return: The `numpy` compatible module for the array `a`.
        """
        pass


_array_types: Dict[type, ArrayType] = dict()


def _register_array_type(ty, get_memory_impl: ArrayType):
    _array_types[ty] = get_memory_impl


def can_assign_from(a, b):
    """
    :param a: An array.
    :param b: An array.
    :return: True iff `a` supports assignments from `b`.
    """
    return _array_types[type(a)].can_assign_from(a, b)


def get_array_module(a):
    """
    :param a: A numpy-compatible array.
    :return: The numpy-compatible module associated with the array class (e.g., cupy or numpy).
    """
    return _array_types[type(a)].get_array_module(a)


def is_array(a) -> bool:
    """
    :param a: A value.
    :return: True if `a` is an array of some type known to parla.
    """
    return type(a) in _array_types


def asnumpy(a):
    ar = get_array_module(a)
    if hasattr(ar, "asnumpy"):
        return ar.asnumpy(a)
    else:
        return np.asarray(a)


def get_memory(a) -> Memory:
    """
    :param a: An array object.
    :return: A memory in which `a` is stored.
    (Currently multiple memories may be equivalent, because they are associated with CPUs on the same NUMA node,
    for instance, in which case this will return one of the equivalent memories, but not necessarily the one used
    to create the array.)
    """
    if not is_array(a):
        raise TypeError("Array required, given value of type {}".format(type(a)))
    return _array_types[type(a)].get_memory(a)


def copy(destination, source):
    """
    Copy the contents of `source` into `destination`.

    :param destination: The array to write into.
    :param source: The array to read from or the scalar value to put in destination.
    """
    try:
        if is_array(source):
            if can_assign_from(destination, source):
                logger.debug("Direct assign from %r to %r", get_memory(source), get_memory(destination))
                destination[:] = source
            else:
                logger.debug("Copy then assign from %r to %r", get_memory(source), get_memory(destination))
                destination[:] = get_memory(destination)(source)
        else:
            # We assume all non-array types are by-value and hence already exist in the Python interpreter
            # and don't need to be copied.
            destination[:] = source
    except ValueError:
        raise ValueError("Failed to copy from {} to {} ({} {} to {} {})".format(get_memory(source), get_memory(destination),
                                                                          source, getattr(source, "shape", None),
                                                                          destination, getattr(destination, "shape", None)))


def clone_here(source, kind=None):
    """
    Create a local copy of `source` stored at the current location.

    :param source: The array to read from.
    """
    if is_array(source):
        # TODO: How to correctly handle multiple devices.
        return get_current_devices()[0].memory(kind)(source)
    else:
        raise TypeError("Array required, given value of type {}".format(type(source)))


def storage_size(*arrays):
    """
    :return: the total size of the arrays passed as arguments.
    """
    return sum(a.size * a.itemsize for a in arrays)

"""
A hack to deal with concurrent to same stream during data movement
"""
def atomic_copy(destination,original_copy, source):
    try:
        if is_array(source):
            if can_assign_from(destination, source):
                logger.debug("Direct assign from %r to %r", get_memory(source), get_memory(destination))
                for i in range(len(source)):
                    if not original_copy[i] == source[i]:
                        destination[i]=source[i]
            else:
                logger.debug("Copy then assign from %r to %r", get_memory(source), get_memory(destination))
                source = get_memory(destination)(source)
                for i in range(len(source)):
                    if not original_copy[i] == source[i]:
                        destination[i]=source[i]
        else:
            # We assume all non-array types are by-value and hence already exist in the Python interpreter
            # and don't need to be copied.
            destination[:] = source
    except ValueError:
        raise ValueError("Failed to copy from {} to {} ({} {} to {} {})".format(get_memory(source), get_memory(destination),
                                                                          source, getattr(source, "shape", None),
                                                                          destination, getattr(destination, "shape", None)))



class LocalArray:
    def __init__(self, default, kind=None):
        self.default = default
        self.default_copy = cp.deepcopy(default)
        cur_device = get_current_devices()[0]
        local_copy = cur_device.memory(kind)(default)
        self.repr = {cur_device:local_copy}

    def get_copy(self, kind=None):
        device = get_current_devices()[0]
        if device in self.repr.keys():
            source = self.default

            destination = self.repr[device]
            if can_assign_from(destination, source):
                logger.debug("Direct assign from %r to %r", get_memory(source), get_memory(destination))
                destination[:] = source
            else:
                logger.debug("Copy then assign from %r to %r", get_memory(source), get_memory(destination))
            destination[:] = get_memory(destination)(source)


            return  destination
        else:
            #copy data to local device
            source = self.default
            local_copy = device.memory(kind)(source)
            self.repr[device]=local_copy
            #self.cur_device = device
            return  local_copy

    def update_default(self,device):
        cur_update = self.repr[device]
        if can_assign_from(self.default, cur_update):
            logger.debug("Direct update from %r to %r", get_memory(cur_update), get_memory(self.default))
            for i in range(len(cur_update)):
                if not cur_update[i]==self.default_copy[i]:
                    self.default[i] = cur_update[i]
        else:
            logger.debug("Copy then update from %r to %r", get_memory(cur_update), get_memory(self.default))
            cur_update = get_memory(self.default)(cur_update)
            for i in range(len(cur_update)):
                if not cur_update[i]==self.default_copy[i]:
                    self.default[i] = cur_update[i]

_Array_Set = []


def get_device_array(source):
    for a in _Array_Set:
        if (a.default==source).all():
            return  a
    new_array = LocalArray(source)
    _Array_Set.append(new_array)
    return new_array
