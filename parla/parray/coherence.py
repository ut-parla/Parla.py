from __future__ import annotations

from typing import List

import threading

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
    def load(dst, src) -> MemoryOperation:
        """ load data from src to dst """
        return MemoryOperation(MemoryOperation.LOAD, dst, src)

    @staticmethod
    def evict(src) -> MemoryOperation:
        """ invalidate the data in src """
        return MemoryOperation(MemoryOperation.EVICT, src=src)

    @staticmethod
    def check_data(src) -> MemoryOperation:
        """ check if the data is ready, wait if not """
        return MemoryOperation(MemoryOperation.CHECK_DATA, src=src)


class Coherence:
    """
    Implements MSI protocol.
    """
    INVALID = 0
    SHARED = 1
    MODIFIED = 2

    def __init__(self, init_owner: int, num_gpu: int):
        """
        Args:
            init_owner: the owner of the first copy in the system
            num_gpu: number of GPU devices in the system
        """
        self._local_states = {n: self.INVALID for n in range(num_gpu)}  # init GPU status
        self._local_states[CPU_INDEX] = self.INVALID  # init CPU status

        self._local_states[init_owner] = self.SHARED  # update states of current device
        self._owner = init_owner  # owner id when MODIFIED / smallest valid device id when SHARED
        self._global_state = self.SHARED  # state of the whole system

        # is data ready to use in this device?
        # data is not ready if it need to be copied from somewhere (has an load operations in progress)
        # and becomes ready is no data movement in progress
        self._data_ready = {n: True for n in range(num_gpu)}
        self._data_ready[CPU_INDEX] = True

        # held the lock when updating states
        self._lock = threading.Lock()

    def data_is_ready(self, device_id):
        """
        Return True if data on `device_id` is ready to use, and there is no copy in progress.
        """
        return self._data_ready[device_id]

    def set_data_as_ready(self, device_id):
        """
        Mark data on `device_id` as ready to use, and there is no copy in progress.
        """
        self._data_ready[device_id] = True

    def read(self, device_id: int) -> MemoryOperation:
        """ Tell the protocol that this device read from the copy.

        Args:
            device_id: id of this device

        Return:
            MemoryOperation, which is a load operation with src and dst

        Note: lock will be acquired
        """
        with self._lock:
            device_local_state = self._local_states[device_id]

            if device_local_state == self.INVALID:  # need to load data from somewhere
                if self._global_state == self.SHARED:
                    # load data from owner
                    self._local_states[device_id] = self.SHARED

                    self._data_ready[device_id] = False  # copy in progress
                    return MemoryOperation.load(dst=device_id, src=self._owner)
                elif self._global_state == self.MODIFIED:
                    prev_owner = self._owner
                    self._local_states[prev_owner] = self.SHARED
                    self._local_states[device_id] = self.SHARED
                    self._global_state = self.SHARED

                    # Trick: smaller one becomes owner, so will always load from CPU (-1) when possible
                    self._owner = max(self._owner, device_id)


                    self._data_ready[device_id] = False
                    return MemoryOperation.load(dst=device_id, src=prev_owner)
                else:   # overall_state should not be INVALID here
                    return MemoryOperation.error()
            else:
                return MemoryOperation.check_data(device_id)  # check if the data is ready

    def write(self, device_id: int) -> List[MemoryOperation]:
        """ Tell the protocol that this device write to the copy.

        Args:
            device_id: id of this device

        Return:
            List[MemoryOperation], different to _read, write could return several MemoryOperations.
                And the order operations matter.
        Note: lock will be acquired
        """
        with self._lock:
            device_local_state = self._local_states[device_id]

            if device_local_state == self.INVALID:  # need to load data from somewhere
                if self._global_state == self.SHARED:
                    # load data from previous owner
                    prev_owner = self._owner

                    self._data_ready[device_id] = False
                    operations = [MemoryOperation.load(dst=device_id, src=prev_owner)]

                    # evict data from other devices
                    for device, state in self._local_states.items():
                        if state == self.SHARED:
                            self._local_states[device] = self.INVALID
                            operations.append(MemoryOperation.evict(device))

                    # update this device state
                    self._global_state = self.MODIFIED
                    self._local_states[device_id] = self.MODIFIED
                    self._owner = device_id

                    return operations
                elif self._global_state == self.MODIFIED:
                    # load data from previous owner
                    prev_owner = self._owner
                    self._data_ready[device_id] = False
                    operations = [MemoryOperation.load(dst=device_id, src=prev_owner)]

                    # evict data from previous owner
                    self._local_states[prev_owner] = self.INVALID
                    operations.append(MemoryOperation.evict(prev_owner))

                    # update this device state
                    self._global_state = self.MODIFIED
                    self._local_states[device_id] = self.MODIFIED
                    self._owner = device_id

                    return operations
                else:   # overall_state should not be INVALID here
                    return [MemoryOperation.error()]
            elif device_local_state == self.SHARED:  # already have the latest copy
                operations = []

                # evict data from other devices
                for device, state in self._local_states.items():
                    if state == self.SHARED and device != device_id:  # should not include this device itself
                        self._local_states[device] = self.INVALID
                        operations.append(MemoryOperation.evict(device))

                # update this device state
                self._global_state = self.MODIFIED
                self._local_states[device_id] = self.MODIFIED
                self._owner = device_id

                return operations
            else: # this device is the owner in MODIFIED state
                return [MemoryOperation.check_data(device_id)] # do nothing

    def evict(self, device_id: int) -> MemoryOperation:
        """ Tell the protocol that this device want to clear the copy.

        Args:
            device_id: id of this device

        Return:
            MemoryOperation, so the caller could move data following the operation

        Note: if this device has the last copy, the whole protocol state will be INVALID then.
            And the system will lose the copy. So careful when evict the last copy.
        Note: lock will be acquired
        """
        with self._lock:
            device_local_state = self._local_states[device_id]

            if device_local_state == self.INVALID: # already evicted, do nothing
                return MemoryOperation.noop()
            elif device_local_state == self.SHARED:
                # find a new owner
                if device_id == self._owner:
                    new_owner = None
                    for device, state in self._local_states.items():
                        if state == self.SHARED and device != device_id:  # should not include this device itself
                            new_owner = device
                            break
                    if new_owner is None:  # this device owns the last copy
                        self._global_state = self.INVALID  # the system lose the last copy
                    self._owner = new_owner

                # update states
                self._local_states[device_id] = self.INVALID
                return MemoryOperation.evict(device_id)
            else:  # Modified
                self._global_state = self.INVALID  # the system lose the last copy
                self._local_states[device_id] = self.INVALID
                self._owner = None
                return MemoryOperation.evict(device_id)
