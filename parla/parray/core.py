from __future__ import annotations # To support self references in type checking, must be first line

from parla.cpu_impl import cpu
from parla.tasks import get_current_devices
from parla.device import Device

from .coherence import MemoryOperation, Coherence, CPU_INDEX

import numpy
try:  # if the system has no GPU
    import cupy
    from parla.cuda import gpu
except (ImportError, AttributeError):
    # PArray only considers numpy or cupy array
    # work around of checking cupy.ndarray when cupy could not be imported
    cupy = numpy
    gpu = None


class PArray:
    """Multi-dimensional array on a CPU or CUDA device.

    This class is a wrapper around :class:`numpy.ndarray` and :class:`cupy.ndarray`,
    It is used to support Parla sheduler optimization and automatic data movement.

    Args:
        array: :class:`cupy.ndarray` or :class:`numpy.array` object

    Note: some methods are not allowed to be called outside of a Parla task context
    """
    def __init__(self, array) -> None:
        num_devices = gpu.num_devices if gpu else 0

        # _array works as a per device buffer of data
        self._array = {n: None for n in range(num_devices)}  # add gpu id
        self._array[CPU_INDEX] = None  # add cpu id

        # get the array's location
        if isinstance(array, numpy.ndarray):
            location = CPU_INDEX
        else:
            location = int(array.device)

        self._array[location] = array
        self._coherence = Coherence(location, num_devices)  # coherence protocol for managing data among multi device

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

    # Public API:

    def update(self, array) -> None:
        """ Update the copy on current device.

        Args:
            array: :class:`cupy.ndarray` or :class:`numpy.array` object

        Note: should not be called outside of task context
        Note: this method will also update the coherence protocol
        """
        this_device = self._current_device_index

        # check if this array matches the device
        # and copy data to this device
        if isinstance(array, numpy.ndarray):
            if this_device != CPU_INDEX:  # CPU to GPU
                self._array[this_device] = cupy.array(array)
            else: # data already in CPU
                self._array[this_device] = array
        else:
            if this_device == CPU_INDEX: # GPU to CPU
                self._array[this_device] = cupy.asnumpy(array)
            else: # GPU to GPU
                if int(array.device) == this_device: # data already in this device
                    self._array[this_device] = array
                else:  # GPU to GPU
                    self._array[this_device] = cupy.copy(array)

        # update coherence protocol
        operations = self._coherence.update(this_device, do_write=True)
        for op in operations:
            self._process_operation(op, this_device)

    # Coherence update operations:

    def _coherence_read(self, operator: int = None) -> None:
        """ Tell the coherence protocol a read happened on a device.

        And do data movement based on the operations given by protocol.

        Args:
            operator: if is this not None, data will be moved to operator,
                    else move to current device

        Note: should not be called outside of task context
        """
        this_device = self._current_device_index

        if not operator:
            operator = this_device

        # update protocol and get operation
        operation = self._coherence.read(operator)
        self._process_operation(operation, this_device)

    def _coherence_write(self, operator: int = None) -> None:
        """Tell the coherence protocol a write happened on a device.

        And do data movement based on the operations given by protocol.

        Args:
            operator: if is this not None, data will be moved to operator,
                    else move to current device

        Note: should not be called outside of task context
        """
        this_device = self._current_device_index

        if not operator:
            operator = this_device

        # update protocol and get list of operations
        operations = self._coherence.write(operator)
        for op in operations:
            self._process_operation(op, this_device)

    # Device management methods:

    def _process_operation(self, operation: MemoryOperation, current_device: int) -> None:
        """
        Process the given memory operations.
        Data will be moved, and protocol states is kept unchanged.
        """
        if operation.inst == MemoryOperation.ERROR:
            raise RuntimeError(f"PArray gets invalid memory operation from coherence protocol, "
                               f"detail: opcode {operation.inst}, dst {operation.dst}, src {operation.src}")
        elif operation.inst == MemoryOperation.NOOP:
            return  # do nothing
        elif operation.inst == MemoryOperation.LOAD:
            self._copy_data_between_device(operation.dst, operation.src, current_device)
        elif operation.inst == MemoryOperation.EVICT:
            self._array[operation.src] = None  # decrement the reference counter, relying on GC to free the memory
        else:
            raise RuntimeError(f"PArray gets invalid memory operation from coherence protocol, "
                               f"detail: opcode {operation.inst}, dst {operation.dst}, src {operation.src}")

    def _copy_data_between_device(self, dst, src, current_device) -> None:
        """
        Copy data from src to dst.
        #TODO: support P2P copy and Stream
        """
        if src == CPU_INDEX: # copy from CPU to GPU
            if dst == current_device:
                self._array[dst] = cupy.array(self._array[src])
            else:
                with cupy.cuda.Device(dst):
                    self._array[dst] = cupy.array(self._array[src])
        elif dst != CPU_INDEX: # copy from GPU to GPU
            if dst == current_device:
                self._array[dst] = cupy.copy(self._array[src])
            else:
                with cupy.cuda.Device(dst):
                    self._array[dst] = cupy.copy(self._array[src])
        else: # copy from GPU to CPU
            self._array[CPU_INDEX] = cupy.asnumpy(self._array[src])

    @staticmethod
    def _get_current_device() -> Device:
        """
        Get current device from task environment.

        Note: should not be called outside of task context
        """
        return get_current_devices()[0]

    def _auto_move(self, index: int = None, do_write: bool = False) -> None:
        """ Automatically move data to current device.

        Multiple copies on different devices will be made based on coherence protocol.

        Args:
            do_write: True if want make the data writable in coherence protocol
                False if this is only for read only in current task


        Note: should not be called outside of task context
        """
        if do_write:
            self._coherence_write(index)
        else:
            self._coherence_read(index)

    def _on_same_device(self, other: PArray) -> bool:
        """
        Return True if the two PArrays are in the same device.
        Note: other has to be a PArray object.
        """
        this_device = self._current_device_index
        return this_device in other._array and other._array[this_device] is not None

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
        return repr(self._array)

    def __str__(self):
        return str(self._array)

    def __format__(self, format_spec):
        return self._array.__format__(format_spec)
