from __future__ import annotations

from typing import List

CPU_INDEX = -1

class MemoryOperation:
    """
    A memory operation representation.
    """
    ERROR = -1
    NOOP = 0
    LOAD = 1
    EVICT = 2

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

    def update(self, device_id: int, do_write: bool = False) -> List[MemoryOperation]:
        """ Tell the protocol that this device get a new copy (e.g. from user).

        Args:
            device_id: id of the new device
            do_write: if True, this device will be MODIFIED, otherwise SHARED
        """
        device_local_state = self._local_states[device_id]

        if device_local_state == self.MODIFIED:  # already have the right to write data
            return [MemoryOperation.noop()]
        else:  # need to upgrade device state to MODIFIED
            if self._global_state == self.INVALID:  # the system doesn't hold a copy before
                self._local_states[device_id] = self.MODIFIED if do_write else self.SHARED
                self._owner = device_id
                self._global_state = self.MODIFIED if do_write else self.SHARED
                return [MemoryOperation.noop()]
            else:  # the system already hold a copy
                # evict others
                operations = []
                for device, state in self._local_states.items():
                    if state != self.INVALID and device != device_id:  # should not include this device itself
                        self._local_states[device] = self.INVALID
                        operations.append(MemoryOperation.evict(device))

                self._local_states[device_id] = self.MODIFIED if do_write else self.SHARED
                self._owner = device_id
                self._global_state = self.MODIFIED if do_write else self.SHARED
                return operations

    def read(self, device_id: int) -> MemoryOperation:
        """ Tell the protocol that this device read from the copy.

        Args:
            device_id: id of this device

        Return:
            MemoryOperation, which is a load operation with src and dst
        """
        device_local_state = self._local_states[device_id]

        if device_local_state == self.INVALID:  # need to load data from somewhere
            if self._global_state == self.SHARED:
                # load data from owner
                return MemoryOperation.load(dst=device_id, src=self._owner)
            elif self._global_state == self.MODIFIED:
                prev_owner = self._owner
                self._local_states[prev_owner] = self.SHARED
                self._local_states[device_id] = self.SHARED
                self._global_state = self.SHARED

                # Trick: smaller one becomes owner, so will always load from CPU (-1) when possible
                self._owner = min(self._owner, device_id)

                return MemoryOperation.load(dst=device_id, src=prev_owner)
            else:   # overall_state should not be INVALID here
                return MemoryOperation.error()
        else:
            return MemoryOperation.noop()  # do nothing

    def write(self, device_id: int) -> List[MemoryOperation]:
        """ Tell the protocol that this device write to the copy.

        Args:
            device_id: id of this device

        Return:
            List[MemoryOperation], different to _read, write could return several MemoryOperations.
                And the order operations matter.
        """
        device_local_state = self._local_states[device_id]

        if device_local_state == self.INVALID:  # need to load data from somewhere
            if self._global_state == self.SHARED:
                # load data from previous owner
                prev_owner = self._owner
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
            return [MemoryOperation.noop()] # do nothing

    def evict(self, device_id: int) -> MemoryOperation:
        """ Tell the protocol that this device want to clear the copy.

        Args:
            device_id: id of this device

        Return:
            MemoryOperation, so the caller could move data following the operation

        Note: if this device has the last copy, the whole protocol state will be INVALID then.
            And the system will lose the copy. So careful when evict the last copy.
        """
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