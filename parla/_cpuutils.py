import logging

import numpy

from parla import array
from parla.device import Memory, Device, MemoryKind

logger = logging.getLogger(__name__)


class _CPUMemory(Memory):
    @property
    def np(self):
        return numpy

    def __call__(self, target):
        if getattr(target, "device", None) is not None:
            logger.debug("Moving data: %r => CPU", getattr(target, "device", None))
        return array.asnumpy(target)


class _CPUDevice(Device):
    def memory(self, kind: MemoryKind = None):
        return _CPUMemory(self, kind)

    def __repr__(self):
        return "<CPU {} ({}, {}, {})>".format(self.index, self.architecture, self.args, self.kwds)
