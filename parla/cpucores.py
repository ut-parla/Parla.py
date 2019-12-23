import logging

from parla._cpuutils import _CPUDevice, get_n_cores
from .device import Architecture, _register_architecture

__all__ = ["cpu"]

logger = logging.getLogger(__name__)


class _CPUCoresArchitecture(Architecture):
    n_cores: int
    """
    The number of cores for which this process has affinity and are exposed as devices.
    """
    def __init__(self, name, id):
        super().__init__(name, id)
        self.n_cores = get_n_cores()
        self._devices = [self(i) for i in range(self.n_cores)]
        logger.info("CPU 'cores mode' enabled. "
                    "Do not use parallel kernels in this mode (it will cause massive over subscription of the CPU). ")
        logger.info("Parla detected {} cores. Parla cannot currently distinguish threads from core. "
                    "Set CPU affinity to only include one thread on each core to fix this issue.".format(self.n_cores))

    @property
    def devices(self):
        return self._devices

    def __call__(self, id, *args, **kwds):
        return _CPUDevice(self, id, *args, **kwds, n_cores=1)

    def __repr__(self):
        return "CPUCoresArchitecture"

cpu = _CPUCoresArchitecture("CPU Cores", "cpu")
cpu.__doc__ = """The `Architecture` for CPUs.

>>> cpu()
"""

_register_architecture("cpu", cpu)

# Set OpenMP and MKL to use a single thread for calls
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
