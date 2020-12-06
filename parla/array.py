import logging
from abc import ABCMeta, abstractmethod
from typing import Dict
from collections.abc import Sequence

# FIXME: This load of numpy causes problems if numpy is multiloaded. So this breaks using VECs with parla tasks.
#  Loading numpy locally works for some things, but not for the array._register_array_type call.
import numpy as np

from parla.device import Memory
from parla.tasks import get_current_devices

logger = logging.getLogger(__name__)

__all__ = ["get_array_module", "get_memory", "is_array", "asnumpy", "copy", "clone_here", "storage_size",
            "get_device_array"]


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


class LocalArray(Sequence):
    def __init__(self, default):
        self.default = default

    def __getitem__(self,idx):
        ori = self.default[idx]
        if is_array(ori):
            local_data = clone_here(self.default[idx])
            return local_data
        else:
            return ori

    def __setitem__(self,idx, val):
        copy(self.default[idx], val)

    def __len__(self):
        return len(self.default)

    def __repr__(self):
        return "Multi-device-array for {%s}"%(str(self.default))


_Array_Set = []


def get_device_array(source):
    for a in _Array_Set:
        if (a.default == source).all():
            retur a
    new_array = LocalArray(source)
    _Array_Set.append(new_array)
    return new_array
