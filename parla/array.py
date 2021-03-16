import logging
from abc import ABCMeta, abstractmethod
from typing import Dict
import copy as cp
from collections.abc import Sequence
import ast

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
    orig: Any
    device_copy: Dict[tuple, Any]

    def __init__(self, orig):
        logger.debug("wrapping type {}".format(type(orig)))
        self.orig = orig
        if isinstance(orig, list):
            for i in range(len(orig)):
                if isinstance(orig[i], list):
                    self.orig[i] = LocalArray(orig[i])
        #    self.device_copy = cp.deepcopy(orig)
        #else:
        #    self.device_copy = orig.tolist()
        self.device_copy = {}

        # if hasattr(orig, "shape"):
        #     self.shape = orig.shape
        # ...

    def __len__(self):
        return len(self.orig)

    def __getitem__(self, idx: Union[int, slice, tuple]):
        # numpy indexing supports a tuple of slice and integers.
        logger.debug("index type {}".format(type(idx)))
        # 'slice' is unhashable
        tuplize: Dict[type, Callable[[Union[int, slice, tuple]], tuple]] = {
                slice: lambda x: (idx.start, idx.stop, idx.step),
                int: lambda x: (x,),
                tuple: lambda x: x,
                }
        ikey = tuplize[type(idx)](idx)

        if ikey in self.device_copy:
            data = self.device_copy[ikey]
        else:
            data = self.orig[idx]

        if is_array(data):
            local_data = clone_here(data)
            self.device_copy[ikey] = local_data
            return local_data
        else:
            # single element
            return data

    def __setitem__(self, idx, val):
        copy(self.orig[idx], val)

    def __repr__(self):
        return "Multi-device-array for {%s}"%(str(self.orig))


_local_arrays: Dict[int, LocalArray] = {}


def _register_local_array(orig):
    logger.debug("type to be wrapped: {}".format(type(orig)))
    loc = LocalArray(orig)
    _local_arrays[id(orig)] = loc
    return loc


def get_device_array(orig):
    if id(orig) in _local_arrays:
        return _local_arrays[id(orig)]
    return _register_local_array(orig)
