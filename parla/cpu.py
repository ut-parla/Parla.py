import logging
from typing import Dict, Collection

# FIXME: This load of numpy causes problems if numpy is multiloaded. So this breaks using VECs with parla tasks.
#  Loading numpy locally works for some things, but not for the array._register_array_type call.
import numpy

import os
import psutil

from . import array, device
from .array import ArrayType
from .cpu_impl import cpu

__all__ = ["cpu"]


class _NumPyArrayType(ArrayType):
    def can_assign_from(self, a, b):
        return isinstance(b, numpy.ndarray)

    def get_memory(self, a):
        # TODO: This is an issue since there is no way to attach allocations of CPU arrays to specific CPU devices.
        return _CPUMemory(cpu(0))

    def get_array_module(self, a):
        return numpy


device._register_architecture("cpu", cpu)
array._register_array_type(numpy.ndarray, _NumPyArrayType())
