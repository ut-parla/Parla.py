"""
Logical device collections provide a way to map logical devices (and their associates partitions of the data) to
physical devices for execution.
The classes in this module provide tools for mapping 1-d and 2-d arrangements of logical devices onto physical devices.
"""

import functools
import inspect
from abc import abstractmethod, ABCMeta
from functools import reduce
from math import floor, ceil
from typing import List, Tuple, Iterable, Collection, Mapping, Union, Any
from warnings import warn

from parla.device import get_all_devices, Device, Memory, MemoryKind
from parla.tasks import PlacementSource, get_placement_for_any
from parla.array import is_array, copy, clone_here
from parla.warning import PerformanceWarning
from parla.utils import parse_index


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
    """
    A collection of logical devices mapped to physical devices.
    """
    def __init__(self, placement = None):
        """
        :param placement: The physical devices to use or None to use all physical devices.
        """
        devices = get_placement_for_any(placement)
        self._devices = tuple(devices)

    @property
    def devices(self):
        """
        The physical devices used by this collection.
        """
        return self._devices

    @property
    def n_devices(self) -> int:
        """len(self.devices)"""
        return len(self.devices)

    @property
    @abstractmethod
    def n_ldevices(self) -> int:
        pass

    def memory(self, *args: int, kind: MemoryKind = None) -> Memory:
        """
        :param args: The indices of the logical device.
        :param kind: The kind of memory to return.
        :return: The physical memory associated with the specified logical device and memory kind.
        """
        return self.device(*args).memory(kind)

    @abstractmethod
    def device(self, *args: int) -> Device:
        """
        :param args: The indices of the logical device.
        :return: The physical device implementing the specified logical device.
        """
        pass

    @property
    @abstractmethod
    def assignments(self) -> Mapping[Tuple, Device]:
        """
        The mapping from valid indices to the associated physical devices.
        """
        pass


class LDeviceSequence(LDeviceCollection):
    """
    A 1-d collection of logical devices.
    """
    def __init__(self, n_ldevices, placement = None):
        """
        :param n_ldevices: The number of logical devices in this collection.
        :param placement: The physical devices to use or None to use all physical devices.
        """
        super().__init__(placement)
        self._n_ldevices = n_ldevices
        if self.n_ldevices < len(self.devices):
            warn(PerformanceWarning(
                "There are not enough partitions to cover the available devices in mapper: {} < {} (dropping {})"
                .format(self.n_ldevices, len(self.devices), self._devices[:-self.n_ldevices])))
            self._devices = self._devices[-self.n_ldevices:]

    @property
    def n_ldevices(self) -> int:
        return self._n_ldevices

    @abstractmethod
    def device(self, i: int) -> Device:
        """
        :param i: The index of the logical device.
        :return: The physical device.
        """
        pass

    def partition(self, data, memory_kind: MemoryKind = None):
        """
        Partition a dataset over this collection.

        :param data: A function `(int) -> T` where `T` is any type that can be copied by
            `~parla.device.Memory` objects. This function is called for each logical device, passed as an index, to
            get the data for that logical device. The function may also take parameters named `device` and/or `memory`,
            in which case the device and/or memory associated with the logical device is passed along with the index.
        :param memory_kind: The kind of memory in which to place the data.
        :return: A :class:`PartitionedTensor` instance of objects returned by `data` and copied to the appropriate device.
        """
        data = _wrapper_for_partition_function(data)
        return PartitionedTensor([data(i, memory=self.memory(i, kind=memory_kind), device=self.device(i))
                for i in range(self.n_ldevices)])

    def partition_tensor(self, data, overlap=0, memory_kind: MemoryKind = None):
        """
        Partition a tensor along its first dimension, potentially with overlap.

        :param data: A numpy-compatible tensor.
        :param overlap: The number of elements by which partitions should overlap.
        :param memory_kind: The kind of memory in which to store the partitions.
        :return: A :class:`PartitionedTensor` instance of the partition tensors.
        """
        (n, *rest) = data.shape
        return self.partition(lambda i: data[self.slice(i, n, overlap=overlap), ...],
                              memory_kind=memory_kind)

    @abstractmethod
    def slice(self, i: int, n: int, step: int = 1, overlap: int = 0):
        """
        Get a slice object which will select the elements of sequence (of length `n`) which are in partition `i`.

        :param i: The index of the partition.
        :param n: The number of elements to slice (i.e., the length of the sequence this slice will be used on)
        :param step: The step of the slice *within* the partition. If this is non-zero, then the resulting slices
            (for `0 <= i < self.n_ldevices`) will only cover a portion of the values `0 <= j < n`.
        :param overlap: The number of element by which the slices should overlap
            (e.g., the overlap between `i=0` and `i=1`).
        :return: A `slice` object.
        """
        pass

    @property
    def assignments(self):
        return {(i,): self.device(i) for i in range(self.n_ldevices)}


class LDeviceGrid(LDeviceCollection):
    """
    A 2-d collection of logical devices arranged in a grid.
    """
    n_ldevices_x: int
    n_ldevices_y: int

    def __init__(self, n_ldevices_x, n_ldevices_y, placement = None):
        """
        :param n_ldevices_x: The number of logical devices along the 1st dimension of this grid.
        :param n_ldevices_y: The number of logical devices along the 2nd dimension of this grid.
        :param placement: The physical devices to use or None to use all physical devices.
        """
        super().__init__(placement)
        self.n_ldevices_x = n_ldevices_x
        self.n_ldevices_y = n_ldevices_y
        if self.n_ldevices < len(self.devices):
            warn(PerformanceWarning(
                "There are not enough partitions to cover the available devices in mapper: {} < {} (dropping {})"
                .format(self.n_ldevices, len(self.devices), self._devices[:-self.n_ldevices])))
            self._devices = self._devices[:self.n_ldevices]

    @property
    def n_ldevices(self):
        return self.n_ldevices_y * self.n_ldevices_x

    @abstractmethod
    def device(self, i: int, j: int) -> Device:
        """
        :param i: The 1st index of the logical device.
        :param j: The 2nd index of the logical device.
        :return: The physical device.
        """
        pass

    def partition(self, data, memory_kind: MemoryKind = None):
        """
        Partition a dataset over this collection.

        :param data: A function `(int, int) -> T` where `T` is any type that can be copied by
            `~parla.device.Memory` objects. This function is called for each logical device, passed as indices, to
            get the data for that logical device. The function may also take parameters named `device` and/or `memory`,
            in which case the device and/or memory associated with the logical device is passed along with the indices.
        :param memory_kind: The kind of memory in which to place the data.
        :return: A :class:`PartitionedTensor` instance of lists of objects returned by `data` and copied to the appropriate device.
        """
        data = _wrapper_for_partition_function(data)
        return PartitionedTensor([[data(i, j, memory=self.memory(i, j, kind=memory_kind), device=self.device(i, j))
                 for j in range(self.n_ldevices_y)] for i in range(self.n_ldevices_x)])

    def partition_tensor(self, data, overlap=0, memory_kind: MemoryKind = None):
        """
        Partition a tensor in its first two dimension, potentially with overlap.

        :param data: A numpy-compatible tensor.
        :param overlap: The number of elements by which partitions should overlap.
        :param memory_kind: The kind of memory in which to store the partitions.
        :return: A :class:`PartitionedTensor` instance of lists of the partition tensors.
        """
        (n_x, n_y, *rest) = data.shape
        return self.partition(lambda i, j: data[self.slice_x(i, n_x, overlap=overlap),
                                                self.slice_y(j, n_y, overlap=overlap), ...],
                              memory_kind=memory_kind)

    def slice_x(self, i: int, n: int, step: int = 1, *, overlap: int = 0):
        """
        :return: A slice along the 1st dimension of this grid
        :see: `~LDeviceSequence.slice`
        """
        return _partition_slice(i, n, self.n_ldevices_x, overlap=overlap, step=step)

    def slice_y(self, j: int, n: int, step: int = 1, *, overlap: int = 0):
        """
        :return: A slice along the 2st dimension of this grid
        :see: `~LDeviceSequence.slice`
        """
        return _partition_slice(j, n, self.n_ldevices_y, overlap=overlap, step=step)

    @property
    def assignments(self):
        return {(i, j): self.device(i, j) for i in range(self.n_ldevices_x) for j in range(self.n_ldevices_y)}


class LDeviceSequenceBlocked(LDeviceSequence):
    """
    A 1-d collection of logical devices which are assigned to physical devices in contiguous blocks.
    """
    def __init__(self, n_ldevices: int, placement: Union[Collection[PlacementSource], Any, None] = None):
        super().__init__(n_ldevices, placement)
        self._divisor = self.n_ldevices / self.n_devices
        assert floor(self._divisor * self.n_devices) == self.n_ldevices

    def device(self, i):
        if not (0 <= i < self.n_ldevices):
            raise ValueError(i)
        return self.devices[floor(i / self._divisor)]

    def __repr__(self):
        return "{}({})<{}>".format(type(self).__name__, self.n_ldevices, len(self.devices))

    def slice(self, i: int, n: int, step: int = 1, *, overlap: int = 0):
        return _partition_slice(i, n, self.n_ldevices, overlap=overlap, step=step)


class LDeviceGridBlocked(LDeviceGrid):
    """
    A 2-d collection of logical devices which are assigned to physical devices in contiguous blocks in both dimensions.
    """
    def __init__(self, n_ldevices_x: int, n_ldevices_y: int, placement: Union[Collection[PlacementSource], Any, None] = None):
        super().__init__(n_ldevices_x, n_ldevices_y, placement)
        self._n, self._m = _split_number(self.n_devices)
        assert self._n * self._m == self.n_devices
        if self.n_ldevices_x < self._n or self.n_ldevices_y < self._m:
            warn(PerformanceWarning(
                "The logical device grid is not large enough to cover the physical device grid: ({}, {}) < ({}, {})"
                .format(self.n_ldevices_x, self.n_ldevices_y, self._n, self._m)))
        self._divisor_x = self.n_ldevices_x / self._n
        self._divisor_y = self.n_ldevices_y / self._m

    def device(self, i, j):
        if not (0 <= i < self.n_ldevices_x and 0 <= j < self.n_ldevices_y):
            raise ValueError((i, j))
        x = floor(i / self._divisor_x)
        y = floor(j / self._divisor_y)
        return self.devices[(x * self._m) + y]

    def __repr__(self):
        return "{}({}, {})<{}, {}>".format(type(self).__name__, self.n_ldevices_x, self.n_ldevices_y, self._n,
                                           self._m)


class LDeviceGridRaveled(LDeviceGrid):
    """
    A 2-d collection of logical devices which are assigned to physical devices as if `LDeviceSequenceBlocked` were
    applied to a "ravelled" version of the grid of logical devices.
    """
    def __init__(self, n_ldevices_x: int, n_ldevices_y: int, placement: Union[Collection[PlacementSource], Any, None] = None):
        super().__init__(n_ldevices_x, n_ldevices_y, placement)
        self._divisor = self.n_ldevices / self.n_devices

    def device(self, i, j):
        if not (0 <= i < self.n_ldevices_x and 0 <= j < self.n_ldevices_y):
            raise ValueError((i, j))
        return self.devices[floor((i * self.n_ldevices_x + j) / self._divisor)]

    def __repr__(self):
        return "{}({}, {})<{}>".format(type(self).__name__, self.n_ldevices_x, self.n_ldevices_y, self.n_devices)


def _partition_slice(i, n, partitions, step=1, *, overlap=0):
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


IndexType = Union[slice, int, Iterable[int], Tuple[Union[slice, Iterable[int], int]]]


class PartitionedTensor():
    """
    A wrapper of a partitioned tensor.
    """
    def __init__(self, latest_view: List):
        self._latest_view = latest_view

    def __getitem__(self, index: IndexType): # -> Union[Array, List[Array]]
        """
        Read partitions and make sure they are on the current device.

        :param index: index of the target partition(s).

        .. todo:
            Multiple partitions are currently returned as a Python list of partitions (ndarrays).
        """
        if not isinstance(index, tuple):
            index = (index,)
        ret = []
        parse_index(self._latest_view, index, step=lambda I, i: I[i],
                stop=lambda x: ret.append(clone_here(x) if is_array(x) else x))
        if len(ret) == 1:
            return ret[0]
        return ret

    def __setitem__(self, index: IndexType, value):
        """
        Assign :param:`value` to a partition which may not on the current device.

        :param index: index of the target partition(s)

        .. todo:
            Assignment of different values to multiple partitions (ndarrays) are currently NOT supported. The :param:`value` is assigned as a whole to each of the target partition(s).
        """
        if not isinstance(index, tuple):
            index = (index,)
        parse_index(self._latest_view, index, step=lambda I, i: I[i],
                stop=lambda x: copy(x, value))

    def __len__(self):
        return len(self._latest_view)
