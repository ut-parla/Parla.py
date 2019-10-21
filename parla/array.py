import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np

from parla.tasks import get_current_device

logger = logging.getLogger(__name__)

__all__ = ["get_array_module", "get_memory", "is_array", "asnumpy", "copy", "clone_here"]


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


def get_memory(a):
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


def clone_here(source, kind=None):
    """
    Create a local copy of `source` stored at the current location.

    :param source: The array to read from.
    """
    if is_array(source):
        return get_current_device().memory(kind)(source)
    else:
        raise TypeError("Array required, given value of type {}".format(type(source)))

