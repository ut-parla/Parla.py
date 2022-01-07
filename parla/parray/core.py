import numpy
import cupy

from parla.cpu_impl import cpu
from parla.cuda import gpu
from parla.tasks import get_current_devices
from parla.device import Device

class PArray:
    """Multi-dimensional array on a CPU or CUDA device.

    This class is a wrapper around :class:`numpy.ndarray` and :class:`cupy.ndarray`,
    It is used to support Parla sheduler optimization and automatic data movement

    Args:
        array: :class:`cupy.ndarray` or :class:`numpy.array` object
    """
    def __init__(self, array=None) -> None:
        self.array = array

    @property
    def _on_gpu(self) -> bool:
        """
        True if the array is on GPU
        """
        return isinstance(self.array, cupy.ndarray)

    @property
    def _index(self) -> int:
        """
        Index of Current Device
        -1 if on CPU
        """
        if self._on_gpu:
            return self.array.device
        return -1

    @staticmethod
    def _get_current_device() -> Device:
        """
        Get current device from task environment.
        """
        return get_current_devices()[0]

    def _auto_move(self) -> None:
        """
        Automatically move array to current device.
        """
        device = PArray._get_current_device()
        if device.architecture == gpu:
            if self._index == device.index:
                return
            self._to_current_gpu()
        elif device.architecture == cpu:
            self._to_cpu()

    def _to_current_gpu(self) -> None:
        """
        Move the array to current GPU, do nothing if already on the device.
        """
        self.array = cupy.asarray(self.array)  # asarray by default copy to current device

    def _to_gpu(self, index: int) -> None:
        """
        Move the array to GPU, do nothing if already on the device.
        `index` is the index of GPU copied to
        """
        if self._index == index:
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

    def _on_same_device(self, other) -> bool:
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
        if isinstance(ret, numpy.ndarray) or isinstance(ret, cupy.ndarray):
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
