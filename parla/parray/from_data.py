from .core import PArray

import numpy
try:  # if the system has no GPU
    import cupy
except ImportError:
    cupy = numpy  # work around of cupy.ndarray


def array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0, like=None, on_gpu=False):
    """
    Create a Parla array on the specific device (CPU by default).

    Args:
        object: :class:`cupy.ndarray` or :class:`numpy.array` object
            or any other object that can be passed to `numpy.array`.

        dtype: Data type specifier.
        copy (bool): If ``False``, this function returns ``obj`` if possible.
            Otherwise this function always returns a new array.
        order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
            (Fortran-style) order.
            When ``order`` is ``'A'``, it uses ``'F'`` if ``a`` is column-major
            and uses ``'C'`` otherwise.
            And when ``order`` is ``'K'``, it keeps strides as closely as
            possible.
            If ``obj`` is `numpy.ndarray`, the function returns ``'C'``
            or ``'F'`` order array.
            Ignored for cupy(GPU) array.
        subok (bool): If ``True``, then sub-classes will be passed-through,
            otherwise the returned array will be forced to be a base-class
            array (default).
        ndmin (int): Minimum number of dimensions. Ones are inserted to the
            head of the shape if needed.
        like (array_like): Reference object to allow the creation of arrays 
            which are not NumPy arrays. If an array-like passed in as like 
            supports the __array_function__ protocol, the result will be defined by it. 
            In this case, it ensures the creation of an array object compatible with that passed in via this argument.
            New in Numpy version 1.20.0.
            Ignored for cupy(GPU) array.
        on_gpu (bool):
            if ``True``, the new array will be allocated on GPU
            otherwise the new array will be allocated on CPU

    Returns:
        parray.PArray: An array on the current device.
    """
    # if the input is already an ndarray
    if isinstance(object, (numpy.ndarray, cupy.ndarray)):
        if copy:
            parray = PArray(object.copy())
        else:
            parray = PArray(object)
    elif isinstance(object, PArray): # Already an PArray
        if copy:
            parray = PArray(object.array.copy())
        else:
            parray = PArray(object.array)
    else: # create one if it is not an ndarray
        if on_gpu:
            parray = PArray(cupy.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin))
        else:
            if like:
                parray = PArray(numpy.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, like=like))
            else:
                parray = PArray(numpy.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin))
    return parray


def asarray(a, dtype=None, order=None, like=None, on_gpu=False):
    """Converts an object to Parla array.

    This is equivalent to :class:``array(a, dtype, on_gpu, copy=False)``.

    Args:
        a: The source object.
        dtype: Data type specifier. It is inferred from the input by default.
        order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
            (Fortran-style) order.
            When ``order`` is ``'A'``, it uses ``'F'`` if ``a`` is column-major
            and uses ``'C'`` otherwise.
            And when ``order`` is ``'K'``, it keeps strides as closely as
            possible.
            If ``obj`` is `numpy.ndarray`, the function returns ``'C'``
            or ``'F'`` order array.
        like (array_like): Reference object to allow the creation of arrays 
            which are not NumPy arrays. If an array-like passed in as like 
            supports the __array_function__ protocol, the result will be defined by it. 
            In this case, it ensures the creation of an array object compatible with that passed in via this argument.
            New in Numpy version 1.20.0.
            Ignored for cupy(GPU) array.
        on_gpu (bool):
            if ``True``, the new array will be allocated on GPU
            otherwise the new array will be allocated on CPU

    Returns:
        parray.PArray: An array on the current device. If ``a`` is already on
        the device, no copy is performed.

    .. note::
       If ``a`` is an :class:`numpy.ndarray` instance that contains big-endian data,
       this function automatically swaps its byte order to little-endian,
       which is the NVIDIA and AMD GPU architecture's native use.

    .. seealso:: :func:`numpy.asarray`
    """
    return array(a, dtype=dtype, copy=False, order=order, like=like, on_gpu=on_gpu)


def asarray_batch(*args):
    """Converts numpy/cupy ndarray to Parla array without creating additional copy.

    Args:
        ```ndarray(s)```, or ```ndarray(s)``` in ```dict/list/tuple/set``` (could be nested).
        Its structure will be kept.

    Return:
        the same number of Parla array that matches the inputs.

    Example:
        a = numpy.array([1,2])
        b = [cupy.array([3,4]), cupy.array([3,4])]

        a, b = asarray_batch(a, b) # a and b are now parla array
    """
    def get_parray(object):  # recursively process Sequence or Dictionary
        if isinstance(object, (numpy.ndarray, cupy.ndarray)):
            return asarray(object)
        elif isinstance(object, PArray):
            return object
        elif isinstance(object, dict):
            accumulator = {}
            for key, value in object.items():
                accumulator[key] = get_parray(value)
            return accumulator
        elif isinstance(object, (list, tuple, set)):
            accumulator = []
            for item in object:
                accumulator.append(get_parray(item))
            return type(object)(accumulator)
        else:
            raise TypeError(f"Unsupported Type: {type(object)}")
    
    parla_arrays = []
    for arg in args:
        parla_arrays.append(get_parray(arg))
    
    if len(parla_arrays) == 1:
        return parla_arrays[0]  
    else:
        return parla_arrays  # recommend user to unpack this
