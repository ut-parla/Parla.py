from __future__ import annotations

from typing import List

CPU_INDEX = -1

class MemoryOperation:
    """
    A memory operation representation.
    """
    ERROR = -1  # there is an error
    NOOP = 0    # no operation
    LOAD = 1    # load data from src to dst
    EVICT = 2   # clear the data in src

    def __init__(self, inst: int = NOOP, dst: int = -1, src: int = -1):
        self.inst = inst
        self.dst = dst
        self.src = src

    @staticmethod
    def noop() -> MemoryOperation:
        return MemoryOperation()

    @staticmethod
    def error() -> MemoryOperation:
        return MemoryOperation(MemoryOperation.ERROR)

    @staticmethod
    def load(dst, src) -> MemoryOperation:
        return MemoryOperation(MemoryOperation.LOAD, dst, src)

    @staticmethod
    def evict(src) -> MemoryOperation:
        return MemoryOperation(MemoryOperation.EVICT, src=src)


class Coherence:
    """
    A memory coherence protocol between devices.

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
        self._coherence_states = {n: self.INVALID for n in range(num_gpu)}  # init GPU status
        self._coherence_states[CPU_INDEX] = self.INVALID  # init CPU status

        self._coherence_states[init_owner] = self.SHARED  # update states of current device
        self._owner = init_owner  # owner id when MODIFIED / smallest valid device id when SHARED
        self._overall_state = self.SHARED  # state of the whole system

    def update(self, operator: int, do_write: bool =False) -> List[MemoryOperation]:
        """ Tell the protocol that operator get a new copy (e.g. from user).

        Args:
            operator: device id of the new device
            do_write: if True, the operator will be MODIFIED, otherwise SHARED
        """
        operator_state = self._coherence_states[operator]

        if operator_state == self.MODIFIED:  # already have the right to write data
            return [MemoryOperation.noop()]
        else:
            if self._overall_state == self.INVALID:  # the system doesn't hold a copy before
                self._coherence_states[operator] = self.MODIFIED if do_write else self.SHARED
                self._owner = operator
                self._overall_state = self.MODIFIED if do_write else self.SHARED
                return [MemoryOperation.noop()]
            else:  # the system already hold a copy
                # evict others
                operations = []
                for device, state in self._coherence_states.items():
                    if state != self.INVALID and device != operator:  # should not include operator itself
                        self._coherence_states[device] = self.INVALID
                        operations.append(MemoryOperation.evict(device))

                self._coherence_states[operator] = self.MODIFIED if do_write else self.SHARED
                self._owner = operator
                self._overall_state = self.MODIFIED if do_write else self.SHARED
                return operations

    def read(self, operator: int) -> MemoryOperation:
        """ Tell the protocol that operator read from the copy.

        Args:
            operator: device id of the operator

        Return:
            MemoryOperation, so the caller could move data following the operation
        """
        operator_state = self._coherence_states[operator]

        if operator_state == self.INVALID:  # need to load data from somewhere
            if self._overall_state == self.SHARED:
                # load data from owner
                return MemoryOperation.load(dst=operator, src=self._owner)
            elif self._overall_state == self.MODIFIED:
                prev_owner = self._owner
                self._coherence_states[prev_owner] = self.SHARED
                self._coherence_states[operator] = self.SHARED
                self._overall_state = self.SHARED

                # Trick: smaller one becomes owner, so will always load from CPU (-1) when possible
                self._owner = min(self._owner, operator)

                return MemoryOperation.load(dst=operator, src=prev_owner)
            else:   # overall_state should not be INVALID here
                return MemoryOperation.error()
        else:
            return MemoryOperation.noop()  # do nothing

    def write(self, operator: int) -> List[MemoryOperation]:
        """ Tell the protocol that operator write to the copy.

        Args:
            operator: device id of the operator

        Return:
            List[MemoryOperation], different to _read, write could return several MemoryOperations.
                And the order operations matter.
        """
        operator_state = self._coherence_states[operator]

        if operator_state == self.INVALID:  # need to load data from somewhere
            if self._overall_state == self.SHARED:
                # load data from previous owner
                prev_owner = self._owner
                operations = [MemoryOperation.load(dst=operator, src=prev_owner)]

                # evict data from other devices
                for device, state in self._coherence_states.items():
                    if state == self.SHARED:
                        self._coherence_states[device] = self.INVALID
                        operations.append(MemoryOperation.evict(device))

                # update operator state
                self._overall_state = self.MODIFIED
                self._coherence_states[operator] = self.MODIFIED
                self._owner = operator

                return operations
            elif self._overall_state == self.MODIFIED:
                # load data from previous owner
                prev_owner = self._owner
                operations = [MemoryOperation.load(dst=operator, src=prev_owner)]

                # evict data from previous owner
                self._coherence_states[prev_owner] = self.INVALID
                operations.append(MemoryOperation.evict(prev_owner))

                # update operator state
                self._overall_state = self.MODIFIED
                self._coherence_states[operator] = self.MODIFIED
                self._owner = operator

                return operations
            else:   # overall_state should not be INVALID here
                return [MemoryOperation.error()]
        elif operator_state == self.SHARED:  # already have the latest copy
            operations = []

            # evict data from other devices
            for device, state in self._coherence_states.items():
                if state == self.SHARED and device != operator:  # should not include operator itself
                    self._coherence_states[device] = self.INVALID
                    operations.append(MemoryOperation.evict(device))

            # update operator state
            self._overall_state = self.MODIFIED
            self._coherence_states[operator] = self.MODIFIED
            self._owner = operator

            return operations
        else: # operator is the owner in MODIFIED state
            return [MemoryOperation.noop()] # do nothing

    def evict(self, operator: int) -> MemoryOperation:
        """ Tell the protocol that operator want to clear the copy.

        Args:
            operator: device id of the operator

        Return:
            MemoryOperation, so the caller could move data following the operation

        Note: if the operator is the last copy, the whole protocol state will be INVALID then.
            And the system will lose the copy. So careful when evict the last copy.
        """
        operator_state = self._coherence_states[operator]

        if operator_state == self.INVALID: # already evicted, do nothing
            return MemoryOperation.noop()
        elif operator_state == self.SHARED:
            # find a new owner
            if operator == self._owner:
                new_owner = None
                for device, state in self._coherence_states.items():
                    if state == self.SHARED and device != operator:  # should not include operator itself
                        new_owner = device
                        break
                if new_owner is None:  # operator owns the last copy
                    self._overall_state = self.INVALID  # the system lose the last copy
                self._owner = new_owner

            # update states
            self._coherence_states[operator] = self.INVALID
            return MemoryOperation.evict(operator)
        else:  # Modified
            self._overall_state = self.INVALID  # the system lose the last copy
            self._coherence_states[operator] = self.INVALID
            self._owner = None
            return MemoryOperation.evict(operator)