from __future__ import annotations
from typing import List, TYPE_CHECKING, Dict, Union

import threading

if TYPE_CHECKING:  # False at runtime
    from memory import MultiDeviceBuffer
    SlicesType = Union[slice, int, tuple]

CPU_INDEX = -1

class MemoryOperation:
    """
    A memory operation representation.
    """
    ERROR = -1
    NOOP = 0
    LOAD = 1
    EVICT = 2
    CHECK_DATA = 3
    UPDATE_MAP = 4

    inst: int
    dst: int
    src: int

    def __init__(self, inst: int = NOOP, dst: int = -1, src: int = -1):
        self.inst = inst
        self.dst = dst
        self.src = src

    @staticmethod
    def noop() -> MemoryOperation:
        """ no operation """
        return MemoryOperation()

    @staticmethod
    def error() -> MemoryOperation:
        """ there is an error """
        return MemoryOperation(MemoryOperation.ERROR)

    @staticmethod
    def load(dst: int, src: int) -> MemoryOperation:
        """ load data from src to dst """
        return MemoryOperation(MemoryOperation.LOAD, dst, src)

    @staticmethod
    def evict(src: int) -> MemoryOperation:
        """ invalidate the data in src """
        return MemoryOperation(MemoryOperation.EVICT, src=src)

    @staticmethod
    def check_data(src: int) -> MemoryOperation:
        """ check if the data is ready, wait if not """
        return MemoryOperation(MemoryOperation.CHECK_DATA, src=src)

    @staticmethod
    def update_map(src: int) -> MemoryOperation:
        """ update the global/local slices map """
        return MemoryOperation(MemoryOperation.UPDATE_MAP, src=src)


class Coherence:
    """
    Implements fine-grained MSI protocol.

    Each copy could be a subarray of a complete copy
    Assumption: all valid subarray are disjoint
    """
    INVALID = 0
    SHARED = 1
    MODIFIED = 2

    _local_states: Dict[int, int | Dict[int, int]]
    _buffer: MultiDeviceBuffer
    _versions: Dict[int, int | Dict[int, int]]
    _is_complete: Dict[int, bool]
    _data_ready: Dict[int, bool | Dict[int, bool]]
    owner: int
    _global_state: int
    _latest_version: int
    _lock: threading.Lock

    def __init__(self, init_owner: int, num_gpu: int, buffer: MultiDeviceBuffer):
        """
        Args:
            init_owner: the owner of the first copy in the system
            num_gpu: number of GPU devices in the system
        """
        # If copy is complete, value is state
        # if not, value is a Dict{slices_hash: state}
        self._local_states = {n: self.INVALID for n in range(num_gpu)}  # init GPU status
        self._local_states[CPU_INDEX] = self.INVALID                    # init CPU status

        # If copy is complete, value is version
        # if not, value is a Dict{slices_hash: version}
        self._versions = {n: -1 for n in range(num_gpu)}    # init copy version (-1 means no data)
        self._versions[CPU_INDEX] = -1

        # fields used to support fine grained data movement
        self._is_complete = {n: False for n in range(num_gpu)}  # does the device own a complete copy?
        self._is_complete[CPU_INDEX] = False
        self._buffer = buffer                               # underlying buffer corresponding to the protocol

        self._local_states[init_owner] = self.MODIFIED      # initial state is MODIFIED
        self.owner = init_owner                             # the device that has the complete copy (take the role of main memory)
        self._versions[init_owner] = 0                      # the first version is 0
        self._is_complete[init_owner] = True                # the copy is complete
        self._global_state = self.MODIFIED                  # state of the complete system
        self._latest_version = 0                            # the latest version in the system

        # is data ready to use in this device?
        # data is not ready if it need to be copied from somewhere (has an load operations in progress)
        # and becomes ready is no data movement in progress
        # if copy is subarray, value would be a Dict{slices_hash: bool}
        self._data_ready = {n: True for n in range(num_gpu)}
        self._data_ready[CPU_INDEX] = True

        # held the lock when updating states
        self._lock = threading.Lock()

    def data_is_ready(self, device_id: int) -> bool:
        """
        Return True if data on `device_id` is ready to use, and there is no copy in progress.
        """
        if isinstance(self._data_ready[device_id], dict):
            return not (False in self._data_ready[device_id].values())  # all subarray need to be ready
        else:
            return self._data_ready[device_id]

    def set_data_as_ready(self, device_id: int, slices: SlicesType) -> None:
        """
        Mark data on `device_id` as ready to use, and there is no copy in progress.

        Args:
            device_id: id of this device
            slices: slices of the subarray to be manipulated
                    by default equals to None, which means the whole array is manipulated
        """
        if slices: # move a subarray
            slices_hash = self._buffer.get_slices_hash(slices)
            self._data_ready[device_id][slices_hash] = True
        else:
            self._data_ready[device_id] = True

    def _owner_is_latest(self) -> bool:
        """True if owner's has latest version"""
        return self._versions[self.owner] == self._latest_version

    def _write_back_to(self, device_id:int, new_state:int =None) -> List[MemoryOperation]:
        """
        Generate the list of write back MemoryOperation.
        Which make `device_id` has the latest version with a complete copy.

        Args:
            device_id: id of this device
            new_state: new state of `dcvice_id`

        Note: version will be updated
        """
        # write back copies that
        # 1. not INVALID (INVALID copy already write back to owner when it get invalidate)
        # 2. has higher version than owner's
        current_version = self._versions[device_id]
        target = [] # a tuple of (version, device_id)
        latest_complete_copy_id = None
        latest_complete_version = -1

        for id, state in self._local_states.items():
            if isinstance(state, dict):  # subarray need to write back
                for slices_hash, subarray_state in state.items():
                    if subarray_state == self.MODIFIED:
                        target.append(id)

                    if subarray_state != self.INVALID:
                        if new_state == self.MODIFIED:
                            self._local_states[id][slices_hash] = self.INVALID
                        else:
                            self._local_states[id][slices_hash] = self.SHARED
            else:  # write back the latest complete array
                if self._versions[id] >= latest_complete_version and id != device_id:
                    latest_complete_version = self._versions[id]
                    latest_complete_copy_id = id

                if self._local_states[id] != self.INVALID:
                    if new_state == self.MODIFIED:
                        self._local_states[id] = self.INVALID
                    else:
                        self._local_states[id] = self.SHARED

        if latest_complete_copy_id is None:
            raise RuntimeError("There is no valid complete copy")

        if current_version < latest_complete_version:
            target.insert(0, latest_complete_copy_id)

        # update latest version
        if new_state == self.MODIFIED:
            self._local_states[device_id] = self._latest_version + 1
        else:
            self._local_states[device_id] = self._latest_version
        return [MemoryOperation.load(device_id, t) for t in target]

    def read(self, device_id: int, slices: SlicesType) -> List[MemoryOperation]:
        """ Tell the protocol that this device read from the copy.

        Args:
            device_id: id of this device
            slices: slices of the subarray to be manipulated
                         by default equals to None, which means the whole array is manipulated

        Return:
            List[MemoryOperation], which tell how data will be manipulated, and order matter.
            
        Note: lock will be acquired
        """
        operations = []

        with self._lock:
            if slices: # move a subarray
                slices_hash = self._buffer.get_slices_hash(slices)

                if self._is_complete[device_id]:  # use existing complete data at this device
                    device_local_state = self._local_states[device_id]
                else:
                    if not isinstance(self._local_states[device_id], dict):
                        self._versions[device_id] = {}
                        self._local_states[device_id] = {}
                        self._data_ready[device_id] = {}
                        device_local_state = self.INVALID
                    elif slices_hash in self._local_states[device_id]:
                        device_local_state = self._local_states[device_id][slices_hash]
                    else:
                        device_local_state = self.INVALID
                    self._is_complete[device_id] = False  # this is a subarray
            else:
                device_local_state = self._local_states[device_id]
                self._is_complete[device_id] = True  # this is a complete array


            if device_id == self.owner:
                if self._owner_is_latest():
                    operations.append(MemoryOperation.check_data(device_id))  # check if the data is ready
                else:  # update it to latest
                    self._data_ready[self.owner] = False
                    operations.extend(self._write_back_to(self.owner))
            else:
                if device_local_state == self.INVALID:
                    if self._is_complete[device_id]:
                        if self._owner_is_latest():
                            operations.append(MemoryOperation.load(dst=device_id, src=self.owner))
                        else:
                            self._data_ready[self.owner] = False
                            operations.extend(self._write_back_to(self.owner))
                            operations.append(MemoryOperation.load(dst=device_id, src=self.owner))
                    else:  # since we assume all array are disjoint, so could load directly
                        operations.append(MemoryOperation.load(dst=device_id, src=self.owner))
                        operations.append(MemoryOperation.update_map(device_id))
                else:
                    operations.append(MemoryOperation.check_data(device_id))

            # update status
            if self._is_complete[device_id]:
                if device_local_state == self.INVALID:
                    self._local_states[device_id] = self.SHARED
                    self._data_ready[device_id] = False

                self._versions[device_id] = self._versions[self.owner]

                # since are copies are in SHARED states, could pick a new owner
                self.owner = max(self.owner, device_id)
            else:
                if device_local_state == self.INVALID:
                    self._local_states[device_id][slices_hash] = self.SHARED
                    self._data_ready[device_id][slices_hash] = False

                self._versions[device_id][slices_hash] = self._versions[self.owner]

            return operations

    def write(self, device_id: int, slices: SlicesType) -> List[MemoryOperation]:
        """ Tell the protocol that this device write to the copy.

        Args:
            device_id: id of this device
            slices: slices of the subarray to be manipulated
                         by default equals to None, which means the whole array is manipulated

        Return:
            List[MemoryOperation], which tell how data will be manipulated, and order matter.
        """
        operations = []

        with self._lock:
            if slices: # move a subarray
                slices_hash = self._buffer.get_slices_hash(slices)

                if self._is_complete[device_id]:  # use existing complete data at this device
                    device_local_state = self._local_states[device_id]
                else:
                    if not isinstance(self._local_states[device_id], dict):
                        self._versions[device_id] = {}
                        self._local_states[device_id] = {}
                        self._data_ready[device_id] = {}
                        device_local_state = self.INVALID
                    elif slices_hash in self._local_states[device_id]:
                        device_local_state = self._local_states[device_id][slices_hash]
                    else:
                        device_local_state = self.INVALID
                    self._is_complete[device_id] = False  # this is a subarray
            else:
                device_local_state = self._local_states[device_id]
                self._is_complete[device_id] = True  # this is a complete array

            if device_id == self.owner:
                if self._owner_is_latest():
                    operations.append(MemoryOperation.check_data(device_id))  # check if the data is ready
                else:  # update it to latest
                    self._data_ready[self.owner] = False
                    operations.extend(self._write_back_to(self.owner, self.MODIFIED))
            else:
                if device_local_state == self.INVALID:
                    if self._is_complete[device_id]:  # TODO: OPTIMIZE here
                        if self._owner_is_latest():
                            operations.append(MemoryOperation.load(dst=device_id, src=self.owner))
                        else:
                            operations.extend(self._write_back_to(device_id, self.MODIFIED))
                    else:  # since we assume all array are disjoint, so could load directly
                        operations.append(MemoryOperation.load(dst=device_id, src=self.owner))
                        operations.append(MemoryOperation.update_map(device_id))
                else:
                    operations.append(MemoryOperation.check_data(device_id))

            # update status
            if self._is_complete[device_id]:
                if device_local_state == self.INVALID:
                    self._data_ready[device_id] = False

                self._local_states[device_id] = self.MODIFIED
                self._versions[device_id] = self._latest_version + 1

                # since are copies are in SHARED states, could pick a new owner
                self.owner = device_id
            else:
                if device_local_state == self.INVALID:
                    self._data_ready[device_id][slices_hash] = False

                self._local_states[device_id][slices_hash] = self.MODIFIED
                self._latest_version += 1
                self._versions[device_id][slices_hash] = self._latest_version

            return operations


    # def evict(self, device_id: int) -> MemoryOperation:
    #     """ Tell the protocol that this device want to clear the copy.
    #
    #     Args:
    #         device_id: id of this device
    #
    #     Return:
    #         MemoryOperation, so the caller could move data following the operation
    #
    #     Note: if this device has the last copy, the whole protocol state will be INVALID then.
    #         And the system will lose the copy. So careful when evict the last copy.
    #     Note: lock will be acquired
    #     """
    #     with self._lock:
    #         device_local_state = self._local_states[device_id]
    #
    #         if device_local_state == self.INVALID: # already evicted, do nothing
    #             return MemoryOperation.noop()
    #         elif device_local_state == self.SHARED:
    #             # find a new owner
    #             if device_id == self.owner:
    #                 new_owner = None
    #                 for device, state in self._local_states.items():
    #                     if state == self.SHARED and device != device_id:  # should not include this device itself
    #                         new_owner = device
    #                         break
    #                 if new_owner is None:  # this device owns the last copy
    #                     self._global_state = self.INVALID  # the system lose the last copy
    #                 self.owner = new_owner
    #
    #             # update states
    #             self._local_states[device_id] = self.INVALID
    #             return MemoryOperation.evict(device_id)
    #         else:  # Modified
    #             self._global_state = self.INVALID  # the system lose the last copy
    #             self._local_states[device_id] = self.INVALID
    #             self.owner = None
    #             return MemoryOperation.evict(device_id)
