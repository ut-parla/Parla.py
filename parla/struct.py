from __future__ import annotations

from parla.typing import *
from parla.primitives import alignment
from parla import detail

import inspect as pythoninspect

__all__ = [
    "Struct",
    "alignment",
]

class StructMetaclass(type):
    @staticmethod
    def _isDunder(name):
        return name.startswith("__") and name.endswith("__")

    def __new__(cls, name, bases, namespace, **kwds):
        if not bases and name == "Struct":
            # This is the declaration of Struct itself
            return super().__new__(cls, name, bases, namespace, **kwds)

        # if len(bases) != 1 or bases[0].__name__ != "Struct":
        #     raise TypeError("Structures must have the superclass Struct and no others")

        fields = namespace.get("__annotations__", {})

        bad_members = [n for n in namespace.keys() if not cls._isDunder(n) and n not in fields]
        if bad_members:
            raise TypeError("Structs do not support methods or class members: {}".format(", ".join(bad_members)))

        field_order = tuple(fields.keys())
        field_defaults = {n: namespace[n] for n in field_order if n in namespace}
        field_types_dict = {n: eval(fields[n]) for n in field_order}
        field_types = tuple(field_types_dict[n] for n in field_order)

        for n, d in field_defaults.items():
            if not isinstance(d, field_types_dict[n]):
                raise TypeError("The default for {n} has the wrong type {t!s} (expected {r!s})".format(n = n, t = type(d), r = field_types_dict[n]))
                
        # TODO: Slots need to be renamed so that writes can be prevented in immutable cases.
        
        namespace["__slots__"] = field_order
        namespace["__slot_defaults__"] = field_defaults
        namespace["__slot_types__"] = field_types
        for n in field_defaults.keys():
            del namespace[n]

        if "__doc__" not in namespace:
            namespace["__doc__"] = "The structure {name} with fields: {}.".format("; ".join("{} of type {}".format(n, type_to_str(t)) for n, t in field_types_dict.items()), name = name)

        if "__init__" not in namespace:
            def __init__(self, *args, **kwds):
                bases[0].__init__(self, *args, **kwds)
            __init__.__annotations__ = fields
            __init__.__kwdefaults__ = field_defaults
            __init__.__doc__ = "Create a new {name}.".format(name = name)
            params = [
                pythoninspect.Parameter(n, pythoninspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        default = field_defaults.get(n, pythoninspect.Parameter.empty),
                                        annotation = field_types_dict[n])
                for n in field_order]
            __init__.__signature__ = pythoninspect.Signature([pythoninspect.Parameter("self", pythoninspect.Parameter.POSITIONAL_OR_KEYWORD)] + params,
                                                             return_annotation = "{}.Mutable".format(name))
            namespace["__init__"] = __init__

        resultcls = super().__new__(cls, name, bases, namespace, **kwds)

        resultcls.Mutable = resultcls
        
        return resultcls

class Struct(detail.DetailableType, metaclass = StructMetaclass):
    """
    The base class for Parla structures.
    Subclasses should have typed names as the body and no other members.
    Fields can also be assigned a default value.
    (Limitation: All fields with defaults must come after all fields without.)

    .. testsetup::

        from __future__ import annotations
        from parla.struct import *

    >>> class Test(Struct):  # Declare a structure Test
    ...     x : int          # Declare a field x with type int
    ...     y : int = 2      # Declare a field y with a type and a default value
    >>> Test(x = 3, y = 4)
    Test(x=3, y=4)
    >>> Test(3, 4)
    Test(x=3, y=4)
    >>> t = Test(x = 2)
    >>> t
    Test(x=2, y=2)
    >>> t.x
    2

    The main structure type (`Test` above) does not allow fields to be changed.
    Each structure also has a nested subclass `Mutable` (for example, `Test.Mutable`) which also allows fields to be updated using assignment: `test.x = 8`.
    :class:`Alignment<parla.primitives.alignment>` requirements can be given for a `Struct` by using the type `Test.require(alignment(64))`.

    Struct subclasses have simple automatically generated documentation.
    """
    __type_details__ = frozenset((alignment,))
    __slots__ = ()
    
    def __init__(self, *args, **kwds):
        """
        Create an instance of this structure.
        
        The arguments are initial values for the fields of the new structure.
        Positional arguments are in declaration order.
        Any fields not included here use their default value. 
        Fields without a default are required.

        :return: A new mutable structure.
        """
        slots = type(self).__slots__
        typename = type(self).__name__
        bad_kwds = [n for n in kwds.keys() if n not in slots]
        if bad_kwds:
            raise TypeError("{name} only takes keyword arguments {}".format(bad_kwds, name = typename))
        if len(args) > len(slots):
            raise TypeError("{name} only takes at most {} positional arguments".format(len(slots), name = typename))
        for n, v in type(self).__slot_defaults__.items():
            setattr(self, n, v)
        for n, v in zip(slots, args):
            setattr(self, n, v)
            if n in kwds:
                raise TypeError("{n} is given as both a positional and keyword argument".format(n = n))
        for n, v in kwds.items():
            setattr(self, n, v)
        for n, t in zip(slots, type(self).__slot_types__):
            try:
                v = getattr(self, n)
            except AttributeError:
                raise TypeError("No value provided for {n}".format(n = n))                
            if not isinstance(v, t):
                raise TypeError("The value for {n} has the wrong type {t!s} (expected {r!s})".format(n = n, t = type(v), r = t))

    def __repr__(self):
        return "{name}({kwds})".format(name = type(self).__name__, kwds = ", ".join("{}={}".format(n, getattr(self, n)) for n in type(self).__slots__))
