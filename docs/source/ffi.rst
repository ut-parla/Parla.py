Foreign Function Interface
==========================

To call a foreign function from within Parla code, the function needs to be imported.
The import will provide a raw callable which accepts special C-like types.
In general, the raw callable will need to be wrapped to provide an easy to use Parla interface.
This wrapper uses unsafe operations to extract the low-level pointers and structural information from arrays and potentially convert other types.

The :ref:`Parla Prototype` uses Python and Numba, so `CFFI <https://cffi.readthedocs.io/en/latest/>`_ provides foreign function import.
Since it will use Dynd, we will probide utility methods for functions which support Dynd arrays directly and native Dynd operations.

Using these tools, the wrapper (a normal Parla function) for a `gemv` would be:

.. testsetup::

   from __future__ import annotations
   from parla.primitives import *
   from parla.array import *

>>> def sgemv(alpha : F[32], a : Array[F[32], 2], x : Array[F[32], 1], beta : F[32], y : Array[F[32], 1].Mutable) -> Void:
...     # Determine appropriate layout and trans arguments for gemv
...     layout = ...
...     trans = ...
...     cblas_sgemv(layout, trans, 
...        a.size(0), a.size(1), alpha, UNSAFE_data_address(a), UNSAFE_stride(a, 0), 
...        UNSAFE_data_address(x), UNSAFE_stride(x, 0), 
...        beta, UNSAFE_data_address(y), UNSAFE_stride(y, 0))
