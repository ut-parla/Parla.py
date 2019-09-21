import functools
import inspect
from abc import abstractmethod, ABCMeta
from functools import reduce
from math import floor, ceil
from typing import List, Tuple, Collection, Mapping
from warnings import warn

from parla.device import get_all_devices, Device, Memory, MemoryKind
from parla.warning import PerformanceWarning


def _factors(n: int) -> List[int]:
    for m in range(2, ceil(n ** 0.5) + 1):
        if n % m == 0:
            return [m] + _factors(n // m)
    return [n]


def _split_number(n: int) -> Tuple[int, int]:
    f = _factors(n)
    if len(f) == 1:
        f += [1]
    fa, fb = f[:len(f) // 2], f[len(f) // 2:]
    return reduce(int.__mul__, fa), reduce(int.__mul__, fb)


class LDeviceCollection(metaclass=ABCMeta):
    def __init__(self, devices):
        if devices is None:
            devices = get_all_devices()
        self._devices = tuple(devices)

    @property
    def devices(self):
        return self._devices

    @property
    def n_devices(self) -> int:
        return len(self.devices)

    @property
    @abstractmethod
    def n_partitions(self) -> int:
        pass

    def memory(self, *args: int, kind: MemoryKind = None) -> Memory:
        return self.device(*args).memory(kind)

    @abstractmethod
    def device(self, *args: int) -> Device:
        pass

    @property
    @abstractmethod
    def assignments(self) -> Mapping[Tuple, Device]:
        pass


class LDeviceSequence(LDeviceCollection):
    def __init__(self, n_partitions, devices=None):
        super().__init__(devices)
        self._n_partitions = n_partitions
        if self.n_partitions < len(self.devices):
            warn(PerformanceWarning(
                "There are not enough partitions to cover the available devices in mapper: {} < {} (dropping {})"
                .format(self.n_partitions, len(self.devices), self._devices[:-self.n_partitions])))
            self._devices = self._devices[-self.n_partitions:]

    @property
    def n_partitions(self) -> int:
        return self._n_partitions

    # def memory(self, i: int, *, kind: MemoryKind = None) -> Memory:
    #     return super().memory(i, kind=kind)

    @abstractmethod
    def device(self, i: int) -> Device:
        pass

    def partition(self, data, memory_kind: MemoryKind = None):
        data = _wrapper_for_partition_function(data)
        return [data(i, memory=self.memory(i, kind=memory_kind), device=self.device(i))
                for i in range(self.n_partitions)]

    def partition_tensor(self, data, overlap=0, memory_kind: MemoryKind = None):
        (n, *rest) = data.shape
        return self.partition(lambda i: data[self.slice(i, n, overlap=overlap), ...],
                              memory_kind=memory_kind)

    @abstractmethod
    def slice(self, i: int, n: int, step: int = 1, overlap: int = 0):
        """
        Get a slice object which will select the elements of sequence (of length `n`) which are in partition `i`.

        :param i: The index of the partition.
        :param n: The number of elements to slice (i.e., in the collection this slice will be used on)
        :param step: The step of the slice *within* the partition. If this is non-zero, then the resulting slices
            (for `0 <= i < self.n_partitions`) will only cover a portion of the values `0 <= j < n`.
        :param overlap: The number of element by which the slices should overlap.
        :return: A `slice` object.
        """
        pass

    @property
    def assignments(self):
        return {(i,): self.device(i) for i in range(self.n_partitions)}


class LDeviceGrid(LDeviceCollection):
    n_partitions_x: int
    n_partitions_y: int

    def __init__(self, n_partitions_x, n_partitions_y, devices=None):
        super().__init__(devices)
        self.n_partitions_x = n_partitions_x
        self.n_partitions_y = n_partitions_y
        if self.n_partitions < len(self.devices):
            warn(PerformanceWarning(
                "There are not enough partitions to cover the available devices in mapper: {} < {} (dropping {})"
                .format(self.n_partitions, len(self.devices), self._devices[:-self.n_partitions])))
            self._devices = self._devices[:self.n_partitions]

    @property
    def n_partitions(self):
        return self.n_partitions_y * self.n_partitions_x

    # def memory(self, i: int, j: int, *, kind: MemoryKind = None) -> Memory:
    #     return super().memory(i, j, kind=kind)

    @abstractmethod
    def device(self, i: int, j: int) -> Device:
        pass

    def partition(self, data, memory_kind: MemoryKind = None):
        data = _wrapper_for_partition_function(data)
        return [[data(i, j, memory=self.memory(i, j, kind=memory_kind), device=self.device(i, j))
                 for j in range(self.n_partitions_y)] for i in range(self.n_partitions_x)]

    def partition_tensor(self, data, overlap=0, memory_kind: MemoryKind = None):
        (n_x, n_y, *rest) = data.shape
        return self.partition(lambda i, j: data[self.slice_x(i, n_x, overlap=overlap),
                                                self.slice_y(j, n_y, overlap=overlap), ...],
                              memory_kind=memory_kind)

    def slice_x(self, i: int, n: int, step: int = 1, *, overlap: int = 0):
        return partition_slice(i, n, self.n_partitions_x, overlap=overlap, step=step)

    def slice_y(self, j: int, n: int, step: int = 1, *, overlap: int = 0):
        return partition_slice(j, n, self.n_partitions_y, overlap=overlap, step=step)

    @property
    def assignments(self):
        return {(i, j): self.device(i, j) for i in range(self.n_partitions_x) for j in range(self.n_partitions_y)}


class LDeviceSequenceBlocked(LDeviceSequence):
    def __init__(self, n_partitions: int, devices: Collection[Device] = None):
        super().__init__(n_partitions, devices)
        self._divisor = self.n_partitions / self.n_devices
        assert floor(self._divisor * self.n_devices) == self.n_partitions

    def device(self, i):
        if not (0 <= i < self.n_partitions):
            raise ValueError(i)
        return self.devices[floor(i / self._divisor)]

    def __repr__(self):
        return "{}({})<{}>".format(type(self).__name__, self.n_partitions, len(self.devices))

    def slice(self, i: int, n: int, step: int = 1, *, overlap: int = 0):
        return partition_slice(i, n, self.n_partitions, overlap=overlap, step=step)


class LDeviceGridBlocked(LDeviceGrid):
    def __init__(self, n_partitions_x: int, n_partitions_y: int, devices: Collection[Device] = None):
        super().__init__(n_partitions_x, n_partitions_y, devices)
        self._n, self._m = _split_number(self.n_devices)
        assert self._n * self._m == self.n_devices
        if self.n_partitions_x < self._n or self.n_partitions_y < self._m:
            warn(PerformanceWarning(
                "The logical device grid is not large enough to cover the physical device grid: ({}, {}) < ({}, {})"
                .format(self.n_partitions_x, self.n_partitions_y, self._n, self._m)))
        self._divisor_x = self.n_partitions_x / self._n
        self._divisor_y = self.n_partitions_y / self._m

    def device(self, i, j):
        if not (0 <= i < self.n_partitions_x and 0 <= j < self.n_partitions_y):
            raise ValueError((i, j))
        x = floor(i / self._divisor_x)
        y = floor(j / self._divisor_y)
        return self.devices[(x * self._m) + y]

    def __repr__(self):
        return "{}({}, {})<{}, {}>".format(type(self).__name__, self.n_partitions_x, self.n_partitions_y, self._n,
                                           self._m)


class LDeviceGridIrregularlyStripped(LDeviceGrid):
    def __init__(self, n_partitions_x: int, n_partitions_y: int, devices: Collection[Device] = None):
        super().__init__(n_partitions_x, n_partitions_y, devices)
        self._divisor = self.n_partitions / self.n_devices

    def device(self, i, j):
        if not (0 <= i < self.n_partitions_x and 0 <= j < self.n_partitions_y):
            raise ValueError((i, j))
        return self.devices[floor((i * self.n_partitions_x + j) / self._divisor)]

    def __repr__(self):
        return "{}({}, {})<{}>".format(type(self).__name__, self.n_partitions_x, self.n_partitions_y, self.n_devices)


def partition_slice(i, n, partitions, step=1, *, overlap=0):
    partition_size = n / partitions
    return slice(max(0, ceil(i * partition_size) - overlap), min(n, ceil((i + 1) * partition_size) + overlap), step)


def _wrapper_for_partition_function(data):
    # TODO:PERFORMANCE: This could be a performance issue. We may need to use the underlying metadata or
    #  cache the checks for better performance.
    arg_names, args_arg_name, kws_arg_name = inspect.getargs(data.__code__)
    has_memory = "memory" in arg_names
    has_device = "device" in arg_names
    if kws_arg_name is not None or (has_memory and has_device):
        @functools.wraps(data)
        def wrapper(*args, memory, device):
            return memory(data(*args, memory=memory, device=device))
    elif has_memory:
        # noinspection PyUnusedLocal
        @functools.wraps(data)
        def wrapper(*args, memory, device):
            return memory(data(*args, memory=memory))
    elif has_device:
        @functools.wraps(data)
        def wrapper(*args, memory, device):
            return memory(data(*args, device=device))
    else:
        # noinspection PyUnusedLocal
        @functools.wraps(data)
        def wrapper(*args, memory, device):
            return memory(data(*args))
    return wrapper
