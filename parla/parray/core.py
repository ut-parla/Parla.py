from __future__ import annotations

from parla.cpu_impl import cpu
from parla.task_runtime import get_current_devices
from parla.device import Device

from .coherence import MemoryOperation, Coherence, CPU_INDEX
from typing import List

import threading
import numpy
try:  # if the system has no GPU
    import cupy
    num_devices = cupy.cuda.runtime.getDeviceCount()
except (ImportError):
    # PArray only considers numpy or cupy array
    # work around of checking cupy.ndarray when cupy could not be imported
    cupy = numpy
    num_devices = 0


class PArray:
    """Multi-dimensional array on a CPU or CUDA device.

    This class is a wrapper around :class:`numpy.ndarray` and :class:`cupy.ndarray`,
    It is used to support Parla sheduler optimization and automatic data movement.

    Args:
        array: :class:`cupy.ndarray` or :class:`numpy.array` object

    Note: some methods should be called within the current task context
    """
    def __init__(self, array) -> None:
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

        # a condition variable to acquire when moving data on the device
        self._coherence_cv = {n:threading.Condition() for n in range(num_devices)}
        self._coherence_cv[CPU_INDEX] = threading.Condition()

    # Properties:

    @property
    def array(self):
        """
        The reference to cupy/numpy array on current device.
        Note: should be called within the current task context
        """
        return self._array[self._current_device_index]

    @property
    def _on_gpu(self) -> bool:
        """
        True if the array is on GPU.
        Note: should be called within the current task context
        """
        return self._current_device_index != CPU_INDEX

    @property
    def _current_device_index(self) -> int:
        """
        -1 if the current device is CPU.
        Otherwise GPU ID.
        Note: should be called within the current task context
        """
        device = PArray._get_current_device()
        if device.architecture == cpu:
            return CPU_INDEX
        else:
            # assume GPU here, won't check device.architecture == gpu
            # to avoid import `gpu`, which is slow to setup.
            return device.index

            # Public API:

    def update(self, array) -> None:
        """ Update the copy on current device.

        Args:
            array: :class:`cupy.ndarray` or :class:`numpy.array` object

        Note: should be called within the current task context
        Note: data should be put in OUT/INOUT fields of spawn
        """
        this_device = self._current_device_index

        if isinstance(array, numpy.ndarray):
            if this_device != CPU_INDEX:  # CPU to GPU
                self._array[this_device] = cupy.asarray(array)
            else: # data already in CPU
                self._array[this_device] = array
        else:
            if this_device == CPU_INDEX: # GPU to CPU
                self._array[this_device] = cupy.asnumpy(array)
            else: # GPU to GPU
                if int(array.device) == this_device: # data already in this device
                    self._array[this_device] = array
                else:  # GPU to GPU
                    dst_data = cupy.empty_like(array)
                    dst_data.data.copy_from_device_async(array.data, array.nbytes)
                    self._array[this_device] = dst_data

    # Coherence update operations:

    def _coherence_read(self, device_id: int = None) -> None:
        """ Tell the coherence protocol a read happened on a device.

        And do data movement based on the operations given by protocol.

        Args:
            device_id: if is this not None, data will be moved to this device,
                    else move to current device

        Note: should be called within the current task context
        """
        if not device_id:
            device_id = self._current_device_index

        # update protocol and get operation
        operation = self._coherence.read(device_id) # locks involve
        self._process_operations([operation]) # condition variable involve

    def _coherence_write(self, device_id: int = None) -> None:
        """Tell the coherence protocol a write happened on a device.

        And do data movement based on the operations given by protocol.

        Args:
            device_id: if is this not None, data will be moved to this device,
                    else move to current device

        Note: should be called within the current task context
        """
        if not device_id:
            device_id = self._current_device_index

        # update protocol and get operation
        operations = self._coherence.write(device_id) # locks involve
        self._process_operations(operations) # condition variable involve

    # Device management methods:

    def _process_operations(self, operations: List[MemoryOperation]) -> None:
        """
        Process the given memory operations.
        Data will be moved, and protocol states is kept unchanged.
        """
        for op in operations:
            if op.inst == MemoryOperation.NOOP:
                pass  # do nothing
            elif op.inst == MemoryOperation.CHECK_DATA:
                if not self._coherence.data_is_ready(op.src):  # if data is not ready, wait
                    with self._coherence_cv[op.src]:
                        while not self._coherence.data_is_ready(op.src):
                            self._coherence_cv[op.src].wait()
            elif op.inst == MemoryOperation.LOAD:
                with self._coherence_cv[op.dst]:  # hold the CV when moving data
                    with self._coherence_cv[op.src]:  # wait on src until it is ready
                        while not self._coherence.data_is_ready(op.src):
                            self._coherence_cv[op.src].wait()
                    self._copy_data_between_device(op.dst, op.src)  # copy data
                    self._coherence.set_data_as_ready(op.dst)  # mark it as done
                    self._coherence_cv[op.dst].notify_all()  # let other threads know the data is ready
            elif op.inst == MemoryOperation.EVICT:
                self._array[op.src] = None  # decrement the reference counter, relying on GC to free the memory
                self._coherence.set_data_as_ready(op.src)  # mark it as done
            elif op.inst == MemoryOperation.ERROR:
                raise RuntimeError("PArray gets an error from coherence protocol")
            else:
                raise RuntimeError(f"PArray gets invalid memory operation from coherence protocol, "
                                   f"detail: opcode {op.inst}, dst {op.dst}, src {op.src}")

    def _copy_data_between_device(self, dst, src) -> None:
        """
        Copy data from src to dst.
        """
        if src == dst:
            return
        elif src == CPU_INDEX: # copy from CPU to GPU
            self._array[dst] = cupy.asarray(self._array[src])
        elif dst != CPU_INDEX: # copy from GPU to GPU
            src_data = self._array[src]
            dst_data = cupy.empty_like(src_data)
            dst_data.data.copy_from_device_async(src_data.data, src_data.nbytes)
            self._array[dst] = dst_data
        else: # copy from GPU to CPU
            self._array[CPU_INDEX] = cupy.asnumpy(self._array[src])

    @staticmethod
    def _get_current_device() -> Device:
        """
        Get current device from task environment.

        Note: should be called within the current task context
        """
        return get_current_devices()[0]

    def _auto_move(self, device_id: int = None, do_write: bool = False) -> None:
        """ Automatically move data to current device.

        Multiple copies on different devices will be made based on coherence protocol.

        Args:
            device_id: current device id. CPU use CPU_INDEX as id
            do_write: True if want make the device MO in coherence protocol
                False if this is only for read only in current task

        Note: should be called within the current task context
        """
        if do_write:
            self._coherence_write(device_id)
        else:
            self._coherence_read(device_id)

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
        if isinstance(ret, numpy.ndarray):
            return PArray(ret)
        elif isinstance(ret, cupy.ndarray):
            if ret.shape == ():
                return ret.item()
            else:
                return PArray(ret)
        else:
            return ret

    def __setitem__(self, slices, value):
        if isinstance(value, PArray):
            self.array.__setitem__(slices, value.array)
        else:
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
