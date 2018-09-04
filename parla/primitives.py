"""
Parla also provides a number of primitive types.

>>> x : I[32]
>>> y : UI[32]
>>> z : F[32]
"""

from __future__ import annotations

from parla.typing import *

__all__ = [
    "Boolean", "I", "UI", "F"
]

Boolean = bool

class _PrimitiveTypeBuilder:
    def __init__(self, name, t, doc):
        self.__doc__ = doc
        self.__name__ = name
        self._type = t
    def __getitem__(self, size):
        if not isinstance(size, int) or size <= 0:
            raise TypeError("Size must be a positive int.")
        return self._type
    def __getattr__(self, attrname):
        return getattr(self._type, attrname)
    
I = _PrimitiveTypeBuilder("I", int, """
Signed Integer.

:usage: I[size]

:param size: The size of the value in *bits*.
    Implementations will probably restrict `size` to a set, such as :math:`\{ 8, 16, 32, 64 \}`.
""")
UI = _PrimitiveTypeBuilder("UI", int, """
Unsigned Integer.

:usage: UI[size]

:param size: The size of the value in *bits*.
    Implementations will probably restrict `size` to a set, such as :math:`\{ 8, 16, 32, 64 \}`.
""")
F = _PrimitiveTypeBuilder("F", float, """
IEEE-like Floating-point.

:usage: F[size]

:param size: The size of the value in *bits*.
    Implementations will probably restrict `size` to a set, such as :math:`\{ 32, 64 \}`.
""")
