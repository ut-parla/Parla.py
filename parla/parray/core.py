from __future__ import annotations # To support self references in type checking, must be first line

from parla.cpu_impl import cpu
from parla.tasks import get_current_devices
from parla.device import Device

from typing import List

import numpy
try:  # if the system has no GPU
    import cupy
    from parla.cuda import gpu
except (ImportError, AttributeError):
    # PArray only considers numpy or cupy array
    # work around of checking cupy.ndarray when cupy could not be imported
    cupy = numpy
    gpu = None


CPU_INDEX = -1


class MemoryOperation:
    """
    A memory operation representation.
    """
    ERROR = -1  # there is an error
    NOOP = 0    # no operation
    LOAD = 1    # load data from src to dst
    EVICT = 2   # clear the data in src

    def __init__(self, operation: int = NOOP, dst: int = -1, src: int = -1):
        self.operation = operation
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

    def __init__(self, init_owner: int):
        """
        Args:
            init_owner: the owner of the first copy in the system
        """
        self._coherence_states = {init_owner: self.SHARED}  # states of each device
        self._owner = init_owner  # owner id when MODIFIED / smallest valid device id when SHARED
        self._overall_state = self.SHARED  # state of the whole system

    def register_device(self, operator):
        """ Register a new device to the protocol.

        Should be called before do other operations as an operator.

        Args:
            operator: device id of the new device
        """
        self._coherence_states[operator] = self.INVALID

    def update(self, operator, do_write=False) -> List[MemoryOperation]:
        """ Tell the protocol that operator get a new copy (e.g. from user).

        Should be called before do other operations as an operator.

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

        The operator should already be registered in the protocol.

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

        The operator should already be registered in the protocol.

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

        The operator should already be registered in the protocol.

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


class PArray:
    """Multi-dimensional array on a CPU or CUDA device.

    This class is a wrapper around :class:`numpy.ndarray` and :class:`cupy.ndarray`,
    It is used to support Parla sheduler optimization and automatic data movement.

    Args:
        array: :class:`cupy.ndarray` or :class:`numpy.array` object

    Note: some methods are not allowed to be called outside of a Parla task context
    """
    def __init__(self, array) -> None:
        self._array = {}
        self._coherence = None

        # get the array's location
        if isinstance(array, numpy.ndarray):
            location = CPU_INDEX
        else:
            location = array.device

        self._array[location] = array
        self._coherence = Coherence(location)

    # Properties:

    @property
    def array(self):
        """
        The reference to cupy/numpy array on current device.
        Note: should not be called outside of task context
        """
        return self._array[self._current_device_index]

    @property
    def _on_gpu(self) -> bool:
        """
        True if the array is on GPU.
        Note: should not be called outside of task context
        """
        return self._current_device_index != CPU_INDEX

    @property
    def _current_device_index(self) -> int:
        """
        -1 if the current device is CPU.
        Otherwise GPU ID.
        Note: should not be called outside of task context
        """
        device = PArray._get_current_device()
        if device.architecture == gpu:
            return device.index
        elif device.architecture == cpu:
            return CPU_INDEX

    # Coherence update operations:

    def _coherence_read(self):
        """
        Tell the coherence protocol a read happened on current device.
        """
        pass


    def _coherence_write(self):
        """
        Tell the coherence protocol a write happened on current device.
        """
        pass

    # Device management methods:

    @staticmethod
    def _get_current_device() -> Device:
        """
        Get current device from task environment.

        Note: should not be called outside of task context
        """
        return get_current_devices()[0]

    def _auto_move(self) -> None:
        """
        Automatically move array to current device.

        Note: should not be called outside of task context
        """
        device = PArray._get_current_device()
        if device.architecture == gpu:
            if self._current_device_index == device.index:
                return
            self._to_current_gpu()
        elif device.architecture == cpu:
            self._to_cpu()

    def _to_current_gpu(self) -> None:
        """
        Move the array to current GPU, do nothing if already on the device.

        Note: should not be called when the system has no GPU
        """
        self.array = cupy.asarray(self.array)  # asarray by default copy to current device

    def _to_gpu(self, index: int) -> None:
        """
        Move the array to GPU, do nothing if already on the device.
        `index` is the index of GPU copied to

        Note: should not be called when the system has no GPU
        """
        if self._current_device_index == index:
            return

        with cupy.cuda.Device(index):
            self.array = cupy.asarray(self.array)

    def _to_cpu(self) -> None:
        """
        Move the array to CPU, do nothing if already on CPU.
        """
        if not self._on_gpu:
            return
        self.array = cupy.asnumpy(self.array)

    def _on_same_device(self, other: PArray) -> bool:
        """
        Return True if the two PArrays are in the same device.
        Note: other has to be a PArray object.
        """
        if self._on_gpu == other._on_gpu:
            return self._on_gpu == False or self.array.device == other.array.device
        return False

    # NumPy/CuPy methods redirection

    def __getattr__(self, item):
        """
        A proxy method that redirect call to methods in :class:`numpy.ndarray` or :class:`cupy.ndarray`
        """
        return getattr(self.array, item)

    # Comparison operators:

    def __lt__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__lt__(y.array)
        else:
            return x.array.__lt__(y)

    def __le__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__le__(y.array)
        else:
            return x.array.__le__(y)

    def __eq__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__eq__(y.array)
        else:
            return x.array.__eq__(y)

    def __ne__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__ne__(y.array)
        else:
            return x.array.__ne__(y)

    def __gt__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__gt__(y.array)
        else:
            return x.array.__gt__(y)

    def __ge__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__ge__(y.array)
        else:
            return x.array.__ge__(y)


    # Truth value of an array (bool):

    def __nonzero__(self):
        return PArray(self.array.__nonzero__())

    # Unary operations:

    def __neg__(self):
        return PArray(self.array.__neg__())

    def __pos__(self):
        return PArray(self.array.__pos__())

    def __abs__(self):
        return PArray(self.array.__abs__())

    def __invert__(self):
        return PArray(self.array.__invert__())

    # Arithmetic:

    def __add__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array + y.array)
        else:
            return PArray(x.array + y)

    def __sub__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array - y.array)
        else:
            return PArray(x.array - y)

    def __mul__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array * y.array)
        else:
            return PArray(x.array * y)

    def __matmul__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array @ y.array)
        else:
            return PArray(x.array @ y)

    def __div__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array / y.array)
        else:
            return PArray(x.array / y)

    def __truediv__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array / y.array)
        else:
            return PArray(x.array / y)

    def __floordiv__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__floordiv__(y.array))
        else:
            return PArray(x.array.__floordiv__(y))

    def __mod__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__mod__(y.array))
        else:
            return PArray(x.array.__mod__(y))

    def __divmod__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__divmod__(y.array))
        else:
            return PArray(x.array.__divmod__(y))

    def __pow__(x, y, modulo):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__pow__(y.array))
        else:
            return PArray(x.array.__pow__(y))

    def __lshift__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__lshift__(y.array))
        else:
            return PArray(x.array.__lshift__(y))

    def __rshift__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__rshift__(y.array))
        else:
            return PArray(x.array.__rshift__(y))

    def __and__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__and__(y.array))
        else:
            return PArray(x.array.__and__(y))

    def __or__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__or__(y.array))
        else:
            return PArray(x.array.__or__(y))

    def __xor__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__xor__(y.array))
        else:
            return PArray(x.array.__xor__(y))

    # Arithmetic, in-place:
    def __iadd__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__iadd__(other.array)
        else:
            self.array.__iadd__(other)
        return self

    def __isub__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__isub__(other.array)
        else:
            self.array.__isub__(other)
        return self

    def __imul__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__imul__(other.array)
        else:
            self.array.__imul__(other)
        return self

    def __idiv__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__idiv__(other.array)
        else:
            self.array.__idiv__(other)
        return self

    def __itruediv__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__itruediv__(other.array)
        else:
            self.array.__itruediv__(other)
        return self

    def __ifloordiv__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ifloordiv__(other.array)
        else:
            self.array.__ifloordiv__(other)
        return self

    def __imod__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__imod__(other.array)
        else:
            self.array.__imod__(other)
        return self

    def __ipow__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ipow__(other.array)
        else:
            self.array.__ipow__(other)
        return self

    def __ilshift__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ilshift__(other.array)
        else:
            self.array.__ilshift__(other)
        return self

    def __irshift__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__irshift__(other.array)
        else:
            self.array.__irshift__(other)
        return self

    def __iand__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__iand__(other.array)
        else:
            self.array.__iand__(other)
        return self

    def __ior__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ior__(other.array)
        else:
            self.array.__ior__(other)
        return self

    def __ixor__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ixor__(other.array)
        else:
            self.array.__ixor__(other)
        return self

    # Container customization:

    def __iter__(self):
        return self.array.__iter__()

    def __len__(self):
        return self.array.__len__()

    def __getitem__(self, slices):
        ret = self.array.__getitem__(slices)

        # ndarray.__getitem__() may return a ndarray
        if isinstance(ret, (numpy.ndarray, cupy.ndarray)):
            return PArray(ret)
        else:
            return ret

    def __setitem__(self, slices, value):
        self.array.__setitem__(slices, value)

    # Conversion:

    def __int__(self):
        return PArray(int(self.array.get()))

    def __float__(self):
        return PArray(float(self.array.get()))

    def __complex__(self):
        return PArray(complex(self.array.get()))

    def __oct__(self):
        return PArray(oct(self.array.get()))

    def __hex__(self):
        return PArray(hex(self.array.get()))

    def __bytes__(self):
        return PArray(bytes(self.array.get()))

    # String representations:

    def __repr__(self):
        return repr(self.array)

    def __str__(self):
        return str(self.array)

    def __format__(self, format_spec):
        return self.array().__format__(format_spec)
