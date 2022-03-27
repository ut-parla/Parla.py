from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Dict, Tuple, Any

import numpy
import cupy

from .coherence import CPU_INDEX

if TYPE_CHECKING:  # False at runtime
    import cupy
    ndarray = Union[numpy.ndarray, cupy.ndarray]
    SlicesType = Union[slice, int, tuple]
    IndicesMapType = List[Union[Dict[int, int], tuple]]

class MultiDeviceBuffer:
    """Underlying Buffer of PArray.

    It holds per device array copy and also index mapping.
    """

    _buffer: Dict[int, ndarray | List[ndarray] | None]
    shape: tuple
    _indices_map: Dict[int, List[IndicesMapType] | None]

    def __init__(self, num_gpu: int):
        # per device buffer
        # key: device_id
        # val: single (complete) ndarray or list of (sub) ndarray
        self._buffer = {n: None for n in range(num_gpu)}  # add gpu id
        self._buffer[CPU_INDEX] = None  # add cpu id

        # per device indices mapping
        # key: device_id
        # val: list of {global_index: local_index} and tuple(begin, end, stop), and the tuple is a represent of slice(begin, end, stop)
        self._indices_map = {n: None for n in range(num_gpu)}
        self._indices_map[CPU_INDEX] = None

        # the shape of the complete array
        self.shape = ()

    def nbytes_at(self, device_id:int) -> int:
        """
        Return the buffer size at `device_id`
        """
        buffer = self._buffer[device_id]
        if buffer is None:
            return 0
        elif isinstance(buffer, list): # subarray at this device buffer
            # size is the sum
            nbytes = 0
            for subarray in buffer:
                nbytes += subarray.nbytes
            return nbytes
        else:  # complete array
            return buffer.nbytes

    def set_complete_array(self, array: ndarray) -> int:
        """
        Add array into the buffer (based on array's device).

        Args:
            array: :class:`cupy.ndarray` or :class:`numpy.array` object

        Return:
            a location (device_id) of the array
        """
        # get the array's location
        if isinstance(array, numpy.ndarray):
            location = CPU_INDEX
        else:
            location = int(array.device)

        self._buffer[location] = array
        self.shape = array.shape
        return location

    def set(self, device_id: int, array: ndarray, is_complete: bool = True, overwrite: bool = False) -> None:
        """
        Set copy at a device, also clean up existing `indices_map` if necessary

        Args:
            device_id: gpu device_id or CPU_INDEX
            array: :class:`cupy.ndarray` or :class:`numpy.array` object
            is_complete: True if `array` is a complete copy, otherwise `array` is a subarray
            overwrite: True if need to clean other subarray copy inside the device before assign the new array
        """
        if is_complete:
            self._indices_map[device_id] = None
            self._buffer[device_id] = array
        else:
            if not isinstance(self._buffer[device_id], List) or overwrite:
                self._indices_map[device_id] = None
                self._buffer[device_id] = [array]
            else:
                self._buffer[device_id].append(array)

    def get(self, device_id: int) -> ndarray | List[ndarray] | None:
        """
        Return the copy at a device

        Args:
            device_id: gpu device_id or CPU_INDEX

        Return
            :class:`cupy.ndarray` or :class:`numpy.array` object
        """
        return self._buffer[device_id]

    def get_global_slices(self, device_id:int, subarray_index:int) -> SlicesType | None:
        """
        Return global slices of one copy at the device.

        If the copy is complete, return None
        """
        if self._indices_map[device_id] is None:
            return None
        else:
            slices = []
            for device_indices in self._indices_map[device_id][subarray_index]:
                if isinstance(device_indices, dict):
                    index = list(device_indices.keys())
                    if len(index) == 1:
                        slices.append(index[0])
                    else:
                        slices.append(index)
                else:
                    slices.append(slice(*device_indices))

            return tuple(slices)

    @staticmethod
    def _map_int_with_int_map(n: int, int_map: Dict[int, int]) -> int | None:
        """
        Find the mapping of `n` in `int_map`

        if `n` not in `int_map`, return None

        example:
            n: 2
            int_map: {1:0, 2:1}
            return: 1
        """
        return None if n not in int_map else int_map[n]

    @staticmethod
    def _map_int_with_slice(n: int, target_slice: tuple) -> int | None:
        """
        Find the mapping of `n` in a `target_slice` (find index of `n` in `target_slice`)
        `target_slice` is a tuple(begin, end, step)

        if `n` not in `target_slice`, return None

        example:
            n: 2
            target_slice: (2, 4, 1)
            return: 0
        """
        # TODO: assume slice is simple (no neg value)
        begin, end, step = target_slice
        step = 1 if step is None else step

        # bound checking
        if n < begin or n >= end:
            return None
        if (n - begin) % step != 0:
            return None

        return (n - begin) // step

    @staticmethod
    def _map_slice_with_slice(input_slice: tuple, target_slice: tuple) -> tuple | None:
        """
        Find the mapping of `input_slice` in a `target_slice`
        `input_slice` and `target_slice` is a tuple(begin, end, step)

        if `input_slice` not a subset of `target_slice`, return None

        example:
            input_slice: (2, 10, 4)
            target_slice: (0, 10, 2)
            return: (1, 5, 2)
        """
        # TODO: assume slice is simple (no neg value)
        target_begin, target_end, target_step = target_slice
        target_step = 1 if target_step is None else target_step

        input_begin, input_end, input_step = input_slice
        input_step = 1 if input_step is None else input_step

        mapped_begin = MultiDeviceBuffer._map_int_with_slice(
            input_begin, target_slice)

        # get the last possible element in range of `input_slice`
        # TODO: what if last_element < input_begin ?
        last_element = input_end - input_step + (input_end - input_begin) % input_step
        mapped_end = MultiDeviceBuffer._map_int_with_slice(last_element, target_slice)

        if mapped_begin is None or mapped_end is None:
            return None

        # adjust step
        if input_step % target_step != 0:
            return None
        mapped_step = input_step // target_step

        return mapped_begin, mapped_end + 1, mapped_step  # tuple

    def map_local_slices(self, device_id: int, global_slices: SlicesType) -> (int, SlicesType):
        """
        Map a given global slices to local slices wrt buffer at the device.

        Raise error if `global_slices` out of range

        Return subarray_index: the index of subarray in the list of `_buffer[device_id]`
               local_slices: the local slices which maps to the `global_slices`
        Note: this method assume a indices mapping exists for this device
        """
        # indexing into the whole array, index of out bound
        not_tuple = False
        if not isinstance(global_slices, tuple):  # if not a tuple, make it a tuple
            global_slices = tuple([global_slices])
            not_tuple = True

        local_slices = []

        if len(self.shape) < len(global_slices):
            raise IndexError(f"index out of range, index:{global_slices}")

        final_subarray_index = 0

        for subarray_index in range(len(self._indices_map[device_id])):  # for each subarray at this device
            indices_map = self._indices_map[device_id][subarray_index]

            for d in range(len(global_slices)):
                size = self.shape[d]  # number of entries at this axis
                global_index = global_slices[d]
                index_map = None if d >= len(indices_map) else indices_map[d]

                if index_map is None:  # None means 1:1 map to all elements at this axis
                    local_index = global_index
                elif isinstance(index_map, dict) and len(index_map) == 1:
                    # special case, this axis was indexed by a int, so
                    # dimension was reduced by 1, 
                    # need to ignore this axis, just check index match or not
                    if list(index_map.keys())[0] == global_index:  # false if type or value doesn't match 
                        continue
                    else:
                        local_index = None
                elif isinstance(index_map, tuple):
                    if isinstance(global_index, int):  # int vs slice
                        local_index = MultiDeviceBuffer._map_int_with_slice(global_index, index_map)
                    elif isinstance(global_index, list):  # List[int] vs slice
                        local_index = [MultiDeviceBuffer._map_int_with_slice(i, index_map) for i in global_index]

                        # any index out of bound?
                        if None in local_index:
                            local_index = None
                    elif isinstance(global_index, slice):  # slice vs slice
                        # slice to tuple
                        slice_tuple = global_index.indices(size)
                        local_tuple = MultiDeviceBuffer._map_slice_with_slice(slice_tuple, index_map)
                        if local_tuple is None:
                            local_index = None
                        else:
                            local_index = slice(*local_tuple)
                    else:
                        raise IndexError(f"Unsupported slices type: {type(global_index)}")
                else:  # Map is int or list<int>
                    if isinstance(global_index, int):  # int vs int/list
                        local_index = self._map_int_with_int_map(global_index, index_map)
                    elif isinstance(global_index, list):  # list vs int/list
                        local_index = [self._map_int_with_int_map(i, index_map) for i in global_index]

                        if None in local_index:
                            local_index = None
                    elif isinstance(global_index, slice):  # slice vs int/list
                        # slice to tuple
                        slice_tuple = global_index.indices(size)
                        local_index = [self._map_int_with_int_map(i, index_map) for i in range(*slice_tuple)]

                        if None in local_index:
                            local_index = None
                    else:
                        raise IndexError(f"Unsupported slices type {type(global_index)}")

                # if None, it means index out of range at this axis
                if local_index is None:
                    # check next copy
                    local_slices = None
                    break

                local_slices.append(local_index)

            if local_slices is None:  # result is not found for this subarray
                if subarray_index == len(self._indices_map[device_id]) - 1:  # this is the last subarray
                    local_slices = None  # non slices is found  
                else: # check next subarray
                    local_slices = []  # clear intermidate result
            else:
                final_subarray_index = subarray_index
                break

        if local_slices is None:
            raise IndexError(f"index out of range, index:{global_slices}")
        elif not_tuple:
            if len(local_slices) == 0:  # only be possible when special case int vs int exists and all axis are ignored
                return final_subarray_index, slice(None, None, None)
            else:
                return final_subarray_index, local_slices[0]
        else:
            return final_subarray_index, tuple(local_slices)

    def set_slices_mapping(self, device_id: int, global_slices: SlicesType):
        """
        set a global slices to local slices mapping wrt buffer at the device.

        Raise error if `global_slices` is higher dim than shape
        Note: this call doesn't check slice is within range, if it is not in range
              exception will be trigger later when trying to index into the copy
        """
        if not isinstance(global_slices, tuple):  # if not a tuple, make it a tuple
            global_slices = tuple([global_slices])

        if len(self.shape) < len(global_slices):
            raise IndexError(f"index out of range, index:{global_slices}")

        slices_map_list = []
        for d in range(len(global_slices)):
            size = self.shape[d]  # number of entries at this axis
            global_slice = global_slices[d]

            if isinstance(global_slice, int):  # a single integer
                slice_map = {global_slice: 0}
            elif isinstance(global_slice, list):  # a list of integer
                slice_map = {global_slice[i]: i for i in range(len(global_slice))}
            elif isinstance(global_slice, slice):  # slice
                # save slice as a tuple
                # None in slice will be instantiated by concrete values
                slice_map = global_slice.indices(size)
            else:
                raise IndexError(f"Unsupported slices type {type(global_slice)}")
            slices_map_list.append(slice_map)

        if self._indices_map[device_id] is None:
            self._indices_map[device_id] = [slices_map_list]
        else:
            self._indices_map[device_id].append(slices_map_list)

    def get_by_global_slices(self, device_id: int, global_slices: SlicesType):
        """
        Indexing/Slicing the buffer by `global_slices`.

        `global_slices` will be first converted into local slices

        Args:
            device_id: gpu device_id or CPU_INDEX
            global_slices: slice/ints/tuple/list<int>, use the same format as advance indexing of numpy

        Return
            :class:`cupy.ndarray` or :class:`numpy.array` object
        """
        # check if there is a mapping
        if self._indices_map[device_id] is None:
            return self._buffer[device_id].__getitem__(global_slices)
        else:
            # map global slices to local slices
            subarray_index, local_slices = self.map_local_slices(device_id, global_slices)
            return self._buffer[device_id][subarray_index].__getitem__(local_slices)

    def set_by_global_slices(self, device_id: int, global_slices: SlicesType, value: ndarray | Any):
        """
        Indexing/Slicing the buffer by `global_slices` and set value.

        `global_slices` will be first converted into local slices

        Args:
            device_id: gpu device_id or CPU_INDEX
            global_slices: slice/ints/tuple/list<int>, use the same format as advance indexing of numpy
            value: the data to set

        Return
            :class:`cupy.ndarray` or :class:`numpy.array` object
        """
        # check if there is a mapping
        if self._indices_map[device_id] is None:
            self._buffer[device_id].__setitem__(global_slices, value)
        else:
            # map global slices to local slices
            subarray_index, local_slices = self.map_local_slices(device_id, global_slices)
            self._buffer[device_id][subarray_index].__setitem__(local_slices, value)


    def _move_data(self, copy_func, dst: int, src: int, subarray_index: int, dst_slices: SlicesType, src_slices: SlicesType, dst_is_current_device:bool = True):
        """
        Helper function for copy_data_between_device
        """
        if dst_is_current_device:
            if dst_slices is None and src_slices is None:  # Complete to Complete
                self._buffer[dst] = copy_func(self._buffer[src])
            elif dst_slices is None and src_slices is not None:  # Incomplete to Complete
                self._buffer[dst][src_slices] = copy_func(self._buffer[src][subarray_index])
            elif dst_slices is not None and src_slices is None:  # Complete to incomplete
                if self._buffer[dst] is None:
                    self._buffer[dst] = []
                self._buffer[dst].append(copy_func(self._buffer[src][dst_slices]))
            else:  # incomplete to incomplete
                raise ValueError("Copy from subarray to subarray is unsupported")
        else:
            with cupy.cuda.Device(dst):  # switch device
                if dst_slices is None and src_slices is None:  # Complete to Complete
                    self._buffer[dst] = copy_func(self._buffer[src])
                elif dst_slices is None and src_slices is not None:  # Incomplete to Complete
                    self._buffer[dst][src_slices] = copy_func(self._buffer[src][subarray_index])
                elif dst_slices is not None and src_slices is None:  # Complete to incomplete
                    if self._buffer[dst] is None:
                        self._buffer[dst] = []
                    self._buffer[dst].append(copy_func(self._buffer[src][dst_slices]))
                else:  # incomplete to incomplete
                    raise ValueError("Copy from subarray to subarray is unsupported")

    def copy_data_between_device(self, dst: int, src: int, dst_is_current_device: bool = True) -> None:
        """
        Copy data from src to dst.

        dst is current device if `dst_is_current_device` is True
        """
        # a function to copy data between GPU devices async
        def copy_from_device_async(src):
            dst_data = cupy.empty_like(src)
            dst_data.data.copy_from_device_async(src.data, src.nbytes)
            return dst_data

        if self._indices_map[src] is None:
            src_slices_list = [None]
        else:
            src_slices_list = [self.get_global_slices(src, i) for i in range(len(self._indices_map[src]))]

        # TRICK: if there are multiple subarray in this device, always pick the last one
        # this is because load of data always comes together with create indices mapping
        # so the indices mapping will put at the end of self._indices_map
        dst_slices = self.get_global_slices(dst, -1)

        for subarray_index in range(len(src_slices_list)):
            src_slices = src_slices_list[subarray_index]
            if src == CPU_INDEX:  # copy from CPU to GPU
                self._move_data(cupy.asarray, dst, src, subarray_index, dst_slices, src_slices, dst_is_current_device)
            elif dst != CPU_INDEX:  # copy from GPU to GPU
                self._move_data(copy_from_device_async, dst, src, subarray_index, dst_slices, src_slices, dst_is_current_device)
            else:  # copy from GPU to CPU
                self._move_data(cupy.asnumpy, dst, src, subarray_index, dst_slices, src_slices)  # dst_is_current_device is no need if dst is CPU

    def get_slices_hash(self, global_slices: SlicesType) -> int:
        """
        Get hash value of a slices of complete array.

        This could be done by replaing list and slice to tuple
        """
        # little chance to have collision, but what if it happened?
        hash_value = 17 # use a none zero hash value, so hash(0) != 0 
        prime = 31
        if not isinstance(global_slices, tuple):
            if isinstance(global_slices, list):
                hash_value = hash_value * prime + hash(tuple(global_slices))
            elif isinstance(global_slices, slice):
                hash_value = hash_value * prime + hash(global_slices.indices(self.shape[0]))
            else:
                hash_value = hash_value * prime + hash(global_slices)
        else:
            if len(self.shape) < len(global_slices):
                raise IndexError(f"index out of range, index:{global_slices}")

            for d in range(len(global_slices)):
                index = global_slices[d]
                if isinstance(index, list):
                    hash_value = hash_value * prime + hash(tuple(index))
                elif isinstance(index, slice):
                    hash_value = hash_value * prime + hash(index.indices(self.shape[d]))
                else:
                    hash_value = hash_value * prime + hash(index)

        return hash_value

    def __str__(self):
        return str(self._buffer)

    def __contains__(self, device_id):
        """
        Return True if there is a copy in this device
        """
        return device_id in self._buffer and self._buffer[device_id] is not None

    def clear(self, device_id) -> None:
        """
        Clear data in device_id
        """
        self._indices_map[device_id] = None
        self._buffer[device_id] = None
