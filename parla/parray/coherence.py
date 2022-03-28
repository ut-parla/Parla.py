from __future__ import annotations
from typing import List, TYPE_CHECKING, Dict, Union

import threading

if TYPE_CHECKING:  # False at runtime
    SlicesType = Union[slice, int, tuple]

CPU_INDEX = -1

class MemoryOperation:
    """
    A memory operation representation.
    """
    inst: int
    dst: int
    src: int

    # OpCode
    ERROR = -1
    NOOP = 0
    LOAD = 1
    EVICT = 2
    CHECK_DATA = 3

    # Flag
    SWITCH_DEVICE_FLAG = 101  # if the flag is set, it means dst is not the current device
    SKIP_SRC_CHECK = 102      # if the flag is set, no need to hold condition variable of src
    LOAD_SUBARRAY = 103       # if the flag is set, it means a subarray of src should be loaded
    ENSURE_IS_COMPLETE = 104  # if the flag is set, check data will also check if the data is complete

    def __init__(self, inst: int = NOOP, dst: int = -1, src: int = -1, flag: int = []):
        self.inst = inst
        self.dst = dst
        self.src = src
        self.flag = flag

    @staticmethod
    def noop() -> MemoryOperation:
        """ no operation """
        return MemoryOperation()

    @staticmethod
    def error() -> MemoryOperation:
        """ there is an error """
        return MemoryOperation(MemoryOperation.ERROR)

    @staticmethod
    def load(dst: int, src: int, on_different_device: bool = False, skip_src_check: bool = False, is_subarray: bool = False) -> MemoryOperation:
        """ load all data from src to dst 
        
        Need to switch device if `on_different_device` is true
        This could known by checking flag = SWITCH_DEVICE_FLAG
        
        Skip checking condition variable of src if `skip_src_check` is set
        This could known by checking flag = MemoryOperation.SKIP_SRC_CHECK

        If `is_subarray` is True, it means a subarray of src will be loaded
        This could known by checking flag = MemoryOperation.LOAD_SUBARRAY
        """
        flag = []
        if is_subarray:
            flag.append(MemoryOperation.LOAD_SUBARRAY)
        if on_different_device:
            flag.append(MemoryOperation.SWITCH_DEVICE_FLAG)
        if skip_src_check:  
            flag.append(MemoryOperation.SKIP_SRC_CHECK)
        
        return MemoryOperation(MemoryOperation.LOAD, dst, src, flag)

    @staticmethod
    def evict(src: int) -> MemoryOperation:
        """ invalidate the data in src """
        return MemoryOperation(MemoryOperation.EVICT, src=src)

    @staticmethod
    def check_data(src: int, ensure_is_complete:bool = False) -> MemoryOperation:
        """ check if the data is ready, wait if not 

        If `ensure_is_complete` is True
        if means need to wait until and data is completed and ready.
        This could known by checking flag = MemoryOperation.ENSURE_IS_COMPLETE
        """
        flag = []
        if ensure_is_complete:
            flag.append(MemoryOperation.ENSURE_IS_COMPLETE)

        return MemoryOperation(MemoryOperation.CHECK_DATA, src=src, flag=flag)


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
    _versions: Dict[int, int | Dict[int, int]]
    _is_complete: Dict[int, bool]
    _data_ready: Dict[int, bool | Dict[int, bool]]
    owner: int
    _latest_version: int
    _lock: threading.Lock

    def __init__(self, init_owner: int, num_gpu: int):
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
        self._is_complete = {n: None for n in range(num_gpu)}  # does the device own a complete copy? None means neither
        self._is_complete[CPU_INDEX] = None

        self._local_states[init_owner] = self.MODIFIED      # initial state is MODIFIED
        self.owner = init_owner                             # the device that has the complete copy (take the role of main memory)
        self._versions[init_owner] = 0                      # the first version is 0
        self._is_complete[init_owner] = True                # the copy is complete
        self._latest_version = 0                            # the latest version in the system

        # is data ready to use in this device?
        # data is not ready if it need to be copied from somewhere (has an load operations in progress)
        # and becomes ready is no data movement in progress
        # if copy is subarray, value would be a Dict{slices_hash: bool}
        # this provide a order when multiple threads are accessing the same data
        # for example, if multiple threads read the same data on the same device,
        # only one of them will need to performance datamovement and other are just wait ready = True
        self._data_ready = {n: True for n in range(num_gpu)}
        self._data_ready[CPU_INDEX] = True

        # held the lock when updating states
        self._lock = threading.Lock()

    def data_is_ready(self, device_id: int, ensure_is_complete: bool = False) -> bool:
        """
        Return True if data on `device_id` is ready to use, and there is no copy in progress.

        If `ensure_is_complete` is True, return false is the data is not complete even if it is ready
        """
        if isinstance(self._data_ready[device_id], dict):  # there are subarrays on this devices
            if ensure_is_complete:
                return False   # data should be complete
            return not (False in self._data_ready[device_id].values())  # all subarray need to be ready
        else:
            return self._data_ready[device_id]

    def set_data_as_ready(self, device_id: int, slices_hash: int = None) -> None:
        """
        Mark data on `device_id` as ready to use, and there is no copy in progress.

        Args:
            device_id: id of this device
            slices_hash: hash code of the slices of the subarray to be manipulated
                         by default equals to None, which means the whole array is manipulated
        """
        if slices_hash is not None: # move a subarray
            self._data_ready[device_id][slices_hash] = True
        else:
            self._data_ready[device_id] = True

    def _owner_is_latest(self) -> bool:
        """True if owner's has latest version"""
        return self._versions[self.owner] == self._latest_version

    def _write_back_to(self, device_id:int, new_state:int, on_different_device:bool = False, 
                       this_device_id: int = None, skip_src_check_id: int = None) -> List[MemoryOperation]:
        """
        Generate the list of write back MemoryOperation.
        Which make `device_id` has the latest version with a complete copy.

        Args:
            device_id: id of this device
            new_state: new state of `dcvice_id`
            on_different_device: True if this device is not current deivce
            this_device_id: if `on_different_device` is True, this means current device ID. If None, ignore
            skip_src_check_id: skip checking this device's src condition variable when it is not None

        Note: version will be updated
        """
        # write back copies that
        # 1. not INVALID (INVALID copy already write back to owner when it get invalidate)
        # 2. has higher version than owner's
        current_version = self._versions[device_id]
        target = set() # a tuple of (version, device_id)
        latest_complete_copy_id = None
        latest_complete_version = -1

        for id, state in self._local_states.items():
            if isinstance(state, dict):  # subarray need to write back
                target.add(id)

                if new_state == self.MODIFIED:
                    # invalidate all subarray
                    self._local_states[id] = self.INVALID
                    self._versions[id] = -1
                    self._is_complete[id] = None
                else:
                    # change all states to SHARED
                    for hash, state in self._local_states[id].items():
                        self._local_states[id][hash] = self.SHARED
            else:  # write back the latest complete array
                if self._versions[id] >= latest_complete_version:
                    latest_complete_version = self._versions[id]
                    latest_complete_copy_id = id

                # downgrade
                elif self._local_states[id] == self.SHARED:
                    if new_state == self.MODIFIED:
                        self._local_states[id] = self.INVALID

        if latest_complete_copy_id is None:
            raise RuntimeError("There is no valid complete copy")

        if new_state == self.MODIFIED:
            evict_list = list(target)

            # `this_device_id` is the device id of the final destionation
            # so should write it back but not evict it
            if this_device_id is not None:
                evict_list.remove(this_device_id)
        else:
            evict_list = []

        if current_version < latest_complete_version:
            target = [latest_complete_copy_id] + list(target)  # complete copy first

        # update latest version
        self._versions[device_id] = self._latest_version
        return [MemoryOperation.load(device_id, t, on_different_device=on_different_device, skip_src_check=(skip_src_check_id == t)) for t in target] \
               + [MemoryOperation.evict(t) for t in evict_list]

    def read(self, device_id: int, slices_hash: int = None) -> List[MemoryOperation]:
        """ Tell the protocol that this device read from the copy.

        Args:
            device_id: id of this device
            slices_hash: hash code of the slices of the subarray to be manipulated
                         by default equals to None, which means the whole array is manipulated

        Return:
            List[MemoryOperation], which tell how data will be manipulated, and order matter.
            
        Note: lock will be acquired
        """
        operations = []

        with self._lock:
            if slices_hash is not None: # move a subarray
                if self._is_complete[device_id] is True:  # use existing complete data at this device
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
            elif self._is_complete[device_id] is False:
                # special case: need a complete copy but there are already subarrays in this deivce
                # writeback this subarrays and then copy complete data from owner

                # write back to owner 
                operations.extend(self._write_back_to(self.owner, self.SHARED, on_different_device=True))
                
                # copy from owner
                operations.append(MemoryOperation.load(device_id, self.owner))

                # update status
                self._data_ready[self.owner] = False

                # if there is on going operations on subarrays, no need to set it as False
                if self.data_is_ready(device_id):
                    self._data_ready[device_id] = False

                self._is_complete[device_id] = True
                self._versions[device_id] = self._versions[self.owner]
                self._local_states[self.owner] = self.SHARED  # owner is updated, so it is in SHARED states
                self._local_states[device_id] = self.SHARED

                # change owner
                self.owner = max(self.owner, device_id)

                # skip the rest code
                return operations
            else:  # move a complete copy and current device has no subarrays
                device_local_state = self._local_states[device_id]
                self._is_complete[device_id] = True  # this is a complete array


            if device_id == self.owner:
                if device_local_state == self.SHARED:
                    operations.append(MemoryOperation.check_data(device_id, ensure_is_complete=True))  # check if the data is ready
                elif device_local_state == self.MODIFIED:
                    operations.append(MemoryOperation.check_data(device_id, ensure_is_complete=True))
                    # no need to check data is ready since assume no overlapping writers
                else:  # update it to latest
                    self._data_ready[device_id] = False
                    operations.extend(self._write_back_to(device_id, self.SHARED))
            else:
                if device_local_state == self.INVALID:
                    if self._is_complete[device_id]:
                        operations.extend(self._write_back_to(device_id, self.SHARED))

                        self._data_ready[device_id] = False

                        # change owner
                        if self._owner_is_latest():
                            self.owner = max(self.owner, device_id)
                        else:
                            self.owner = device_id
                    else:  # since we assume all array are disjoint, so could load directly
                        operations.append(MemoryOperation.load(dst=device_id, src=self.owner, is_subarray=True)) 

                        self._data_ready[device_id][slices_hash] = False
                        self._versions[device_id][slices_hash] = self._versions[self.owner]
                elif device_local_state == self.MODIFIED:
                    operations.append(MemoryOperation.check_data(device_id, ensure_is_complete=self._is_complete[device_id]))
                else:
                    operations.append(MemoryOperation.check_data(device_id, ensure_is_complete=self._is_complete[device_id]))

            # update status
            if self._is_complete[device_id]:
                if device_local_state == self.INVALID:
                    self._local_states[device_id] = self.SHARED
            else:
                if device_local_state == self.INVALID:
                    self._local_states[device_id][slices_hash] = self.SHARED

            return operations

    def write(self, device_id: int, slices_hash: int = None) -> List[MemoryOperation]:
        """ Tell the protocol that this device write to the copy.

        Args:
            device_id: id of this device
            slices_hash: hash code of the slices of the subarray to be manipulated
                         by default equals to None, which means the whole array is manipulated

        Return:
            List[MemoryOperation], which tell how data will be manipulated, and order matter.
        """
        operations = []

        with self._lock:
            if slices_hash is not None: # move a subarray
                if self._is_complete[device_id] is True:  # use existing complete data at this device
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
            elif self._is_complete[device_id] is False:
                # special case: need a complete copy but there are already subarrays in this deivce
                # writeback this subarrays and then copy complete data from owner
                #
                # no need to check this since assume no multiwriters
                # operations.append(MemoryOperation.check_data(device_id))

                # write back to owner 
                operations.extend(self._write_back_to(self.owner, self.MODIFIED, 
                                  on_different_device=True, skip_src_check_id=device_id))
                
                # copy from owner
                operations.append(MemoryOperation.load(device_id, self.owner))

                # update status
                self._data_ready[self.owner] = False
                self._data_ready[device_id] = False  # won't deadlock since `skip_src_check_id` is set

                self._is_complete[device_id] = True
                self._versions[device_id] = self._versions[self.owner]
                self._local_states[device_id] = self.MODIFIED
                self._local_states[self.owner] = self.INVALID  # owner is invalid too

                # change owner
                self.owner = device_id

                # skip the rest code
                return operations
            else:
                device_local_state = self._local_states[device_id]
                self._is_complete[device_id] = True  # this is a complete array

            if device_id == self.owner:
                if device_local_state != self.MODIFIED:
                    operations.extend(self._write_back_to(device_id, self.MODIFIED))

                    self._data_ready[device_id] = False
                    self._latest_version += 1
                    self._versions[device_id] = self._latest_version
                else:
                    operations.append(MemoryOperation.noop())
            else:
                if device_local_state == self.INVALID:
                    if self._is_complete[device_id]:
                        operations.extend(self._write_back_to(device_id, self.MODIFIED))

                        self._data_ready[device_id] = False
                        self._latest_version += 1
                        self._versions[device_id] = self._latest_version

                        # change owner
                        if self._owner_is_latest():
                            self.owner = max(self.owner, device_id)
                        else:
                            self.owner = device_id
                    else:  # since we assume all array are disjoint, so could load directly
                        operations.append(MemoryOperation.load(dst=device_id, src=self.owner, is_subarray=True))

                        self._data_ready[device_id][slices_hash] = False
                        self._versions[device_id][slices_hash] = self._versions[self.owner] + 1
                        if self._owner_is_latest():
                            self._latest_version += 1
                        self._local_states[self.owner] = self.INVALID  # invalidate overlapping copy
                elif device_local_state == self.SHARED:
                    if self._is_complete[device_id]:
                        self._latest_version += 1
                        self._versions[device_id] = self._latest_version

                        # change owner
                        if self._owner_is_latest():
                            self.owner = max(self.owner, device_id)
                        else:
                            self.owner = device_id
                    else:
                        self._versions[device_id][slices_hash] += 1
                        self._latest_version = max(self._latest_version, self._versions[device_id][slices_hash])

                    operations.append(MemoryOperation.check_data(device_id, ensure_is_complete=self._is_complete[device_id]))
                else:
                    operations.append(MemoryOperation.check_data(device_id, ensure_is_complete=self._is_complete[device_id]))

            # update status
            if self._is_complete[device_id]:
                if device_local_state != self.MODIFIED:
                    self._local_states[device_id] = self.MODIFIED
            else:
                if device_local_state != self.MODIFIED:
                    self._local_states[device_id][slices_hash] = self.MODIFIED
            return operations

    def evict(self, device_id: int, keep_one_copy: bool = True) -> List[MemoryOperation]:
        """ Tell the protocol that this device want to clear the copy.

        Args:
            device_id: id of this device
            keep_one_copy: if true, writeback the last copy to CPU

        Return:
            List[MemoryOperation], could return several MemoryOperations.
                And the order operations matter.

        Note: if this device has the last copy and `keep_one_copy` is false, 
            the whole protocol state will be INVALID then.
            And the system will lose the copy. Be careful when evict the last copy.
        """
        device_local_state = self._local_states[device_id]
        operations = []

        if device_local_state == self.INVALID: # already evicted, do nothing
            operations.append(MemoryOperation.noop())
        elif device_local_state == self.SHARED:
            if device_id == self._owner:  # has a chance this is the last copy
                # find new owner
                new_owner = None
                for device, state in self._local_states.items():
                    if state == self.SHARED and device != device_id:  # should not include this device itself
                        new_owner = device
                        break

                # this device owns the last copy
                if new_owner is None:  
                    if keep_one_copy:  
                        if device_id == CPU_INDEX:
                            # the last copy is already at CPU, 
                            # do nothing and skip the rest of the code
                            return [MemoryOperation.noop()]
                        else:
                            # write back the last copy to CPU
                            operations.append(MemoryOperation.load(CPU_INDEX, device_id))

                            # now CPU has exclusive access to the data
                            self._global_state = self.MODIFIED
                            self._local_states[CPU_INDEX] = self.MODIFIED

                            new_owner = CPU_INDEX
                    else:
                        self._global_state = self.INVALID  # the system lose the last copy
                self._owner = new_owner

            # update states
            self._local_states[device_id] = self.INVALID
            operations.append(MemoryOperation.evict(device_id))
        else:  # Modified, this device owns the last copy
            if keep_one_copy:  # write back to CPU
                self._owner = CPU_INDEX
                self._local_states[CPU_INDEX] = self.MODIFIED

                operations.append(MemoryOperation.load(CPU_INDEX, device_id))
            else:
                self._global_state = self.INVALID  # the system lose the last copy
                self._owner = None

            self._local_states[device_id] = self.INVALID
            operations.append(MemoryOperation.evict(device_id))

        return operations
