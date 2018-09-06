"""
Parla also provides a number of primitive types.
Primitive types support specifying the `alignment` detail.

.. testsetup::

   from __future__ import annotations
   from parla.primitives import *

>>> x : I[32]
>>> y : UI[32]
>>> z : F[32].require(alignment(8))

"""

from __future__ import annotations

import types as pytypes

from parla.typing import *
from parla import detail

__all__ = [
    "Boolean", "I", "UI", "F"
]

class alignment(detail.Detail):
    """
    A detail specifying the alignment requirements of this type.
    """
    def __init__(self, bytes):
        """
        :param bytes: The alignment for values of this type when stored in an array or `~parla.struct.Struct`.
        """
        self.bytes = bytes
        self.args = (bytes,)

def _make_detailable_prim_type(name, cls):
    def body(ns):
        ns["__type_details__"] = frozenset((alignment,))
    return pytypes.new_class(name, (cls, detail.DetailableType), exec_body=body)

Boolean = bool # _make_detailable_prim_type("Boolean", bool)

class _PrimitiveTypeBuilder:
    def __init__(self, name, t, doc):
        self.__doc__ = doc
        self.__name__ = name
        self._type = t
    def __getitem__(self, size):
        if not isinstance(size, int) or size <= 0:
            raise TypeError("Size must be a positive int.")
        return _make_detailable_prim_type("{}[{}]".format(self.__name__, size), self._type)
    def __getattr__(self, attrname):
        return getattr(self._type, attrname)

I = _PrimitiveTypeBuilder("I", int, """
Signed Integer.

:usage: I[size]

Python `int` values are implicitly converted to `I[size]`, but explicit conversion is possible with ``I[size](v)`` (where `v` is an integral value of some type).
``I[size](s)`` cal also parse the string `s`.

:param size: The size of the value in *bits*.
    Implementations will probably restrict `size` to a set, such as :math:`\{ 8, 16, 32, 64 \}`.
""")
UI = _PrimitiveTypeBuilder("UI", int, """
Unsigned Integer.

:usage: UI[size]

Python `int` values are implicitly converted to `UI[size]`, but explicit conversion is possible with ``UI[size](v)`` (where `v` is an integral value of some type).
``UI[size](s)`` cal also parse the string `s`.

:param size: The size of the value in *bits*.
    Implementations will probably restrict `size` to a set, such as :math:`\{ 8, 16, 32, 64 \}`.
""")
F = _PrimitiveTypeBuilder("F", float, """
IEEE-like Floating-point.

:usage: F[size]

Python `float` values are implicitly converted to `F[size]`, but explicit conversion is possible with ``F[size](v)`` (where `v` is an floating-point value of some type).
``F[size](s)`` cal also parse the string `s` (without loss of precision, even for `F` types larger than `float`).

:param size: The size of the value in *bits*.
    Implementations will probably restrict `size` to a set, such as :math:`\{ 32, 64 \}`.
""")
