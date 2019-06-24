import numpy as np

try:
    import cupy
except ImportError:
    cupy = None

__all__ = ["get_array_module"]


def get_array_module(a):
    """
    :param a: A numpy-compatible array.
    :return: The module associated with the array class (e.g., cupy or numpy).
    """
    if cupy:
        return cupy.get_array_module(a)
    else:
        return np


def asnumpy(a):
    ar = get_array_module(a)
    if hasattr(ar, "asnumpy"):
        return ar.asnumpy(a)
    else:
        return np.asarray(a)