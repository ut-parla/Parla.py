from __future__ import annotations

from parla.typing import *
from parla import function_decorators

__all__ = [
    "Array", "ImmutableArray", "MutableArray",
    "InplaceArray", "InplaceImmutableArray", "InplaceMutableArray",
    "Ref", "ImmutableRef", "MutableRef",
    "ref", "deref", "zeros", "filled",
    "freeze", "UNSAFE_freeze"
]

T = TypeVar("T")
k = LiftedNaturalVar("k")
s1 = LiftedNaturalVar("s₁")
sk = LiftedNaturalVar("sₖ")
    
class Array(TypeConstructor[T, k]):
    """
    Parla arrays are (generally) references to data stored outside the array value itself.
    Arrays are parameterized on the element type `T` and the dimensionality (rank) `k` (a non-negative integer).

    :usage: Array[T, k]

    >>> x : Array[int, 4]

    This base `Array` class allows reading, but provide no guarentees about the mutability of the array via some other reference.

    :meth:`Array.__getitem__` lifts `side-effect-free <parla.function_decorators.side_effect_free>` methods and operators from `T` to `Array`.
    """

    def __getitem__(self, indexing_expression) -> Array[T, count_slices(indexing_expression)]:
        """
        :param indexing_expression: An indexing expression made up of indicies and slices, and optionally an ellipsis.

        :return: Slice or indexed read
        """
        raise NotImplementedError()

    def __getattr__(self, attrname):
        """
        Get the lifted version of an attribute on `T`.
        All `side-effect-free <parla.function_decorators.side_effect_free>` methods and operators (including pure) on `T` are lifted to `Array[T, k]` as element-wise operations.

        :return: The lifted bound method or `Array` of attribute values.
        """
        try:
            return super().__getattr__(attrname)
        except AttributeError:
            v = ... # TODO: Get the attribute from `T`
            if callable(v) and not function_decorators.has_property(v, function_decorators.side_effect_free):
                raise AttributeError("{} has side-effects and cannot be used on an immutable reference.".format(attrname))
            # TODO: Lift the attribute as needed
            raise NotImplementedError()
            return v
    

class MutableArray(Array):
    """
    A reference to a mutable `Array`.

    :usage: MutableArray[T, k]
    
    :meth:`MutableArray.__getitem__` lifts all methods and operators from `T` to `Array`.
    """
    def __getitem__(self, indexing_expression) -> MutableArray[T, count_slices(indexing_expression)]:
        pass
        
    def __setitem__(self, indexing_expression, a : Array[T, count_slices(indexing_expression)]) -> Void:
        """
        Assign new values to the slice.

        :param indexing_expression: An indexing expression made up of indicies and slices, and optionally an ellipsis.
        :param a: The array to copy into the slice. `a` will be broadcast as needed.
        """
        raise NotImplementedError()

    def __getattr__(self, attrname):
        """
        Get the lifted version of an attribute on `T`.
        *All* methods and operators (including side-effecting) on `T` are lifted to `Array[T, k]` as element-wise operations.

        :return: The lifted bound method or `Array` of attribute values.
        """
        try:
            return super().__getattr__(attrname)
        except AttributeError:
            v = ... # TODO: Get the attribute from `T`
            if callable(v) and not function_decorators.has_property(v, function_decorators.side_effect_free):
                raise AttributeError("{} has side-effects and cannot be used on an immutable reference.".format(attrname))
            # TODO: Lift the attribute as needed
            raise NotImplementedError()
            return v

class ImmutableArray(Array):
    """
    A reference to an *immutable* `Array`.
    The programmer and the runtime can assume no mutable references exist to this array.

    :usage: ImmutableArray[T, k]
    """
    def __getitem__(self, indexing_expression) -> ImmutableArray[T, count_slices(indexing_expression)]:
        raise NotImplementedError()

## Inplace Arrays

class InplaceArray(Array[T, s1, ..., sk], TypeConstructor[T, k]):
    """
    Parla in-place arrays have a fixed shape and store their data along with their metainformation (instead of storing a reference to external data).
    This means that in-place arrays do not result in pointers and are similar to structures with numbered fields, all of the same type.
    `InplaceArray` is parameterized on the element type `T` and the shape `s₁, …, sₖ`.

    :usage: InplaceArray[T, s₁, …, sₖ]

    >>> x : InplaceArray[float, 4, 4]

    Here `x` is a 4 by 4 in-place array of floats which could store a 3-d coordinate transform.

    `InplaceArray` exposes the same API as `Array` and views on `InplaceArray` are `Array` (or the appropriate mutability subtype).
    If `T` has a "zero" then the "zero" of `InplaceArray[T, s₁, …, sₖ]` is the inplace array filled with `T`'s "zero".

    .. todo:: How are InplaceArrays created?
    """
    pass

class InplaceMutableArray(MutableArray, InplaceArray):
    """
    A mutable in-place array.

    :usage: InplaceMutableArray[T, s₁, …, sₖ]

    :see: `InplaceArray`
    """
    pass

class InplaceImmutableArray(ImmutableArray, InplaceArray):
    """
    An immutable in-place array.

    :usage: InplaceImmutableArray[T, s₁, …, sₖ]

    :see: `InplaceArray`
    """
    pass

    
## Refs

class _RefBuilder(GenericClassAlias):
    def __init__(self, name, ArrayCls, doc):
        self.__doc__ = doc
        self.__name__ = name
        self._ArrayCls = ArrayCls
    def __getitem__(self, indexing_expression):
        return self._ArrayCls[indexing_expression, 0]
    def __getattr__(self, attrname):
        return getattr(self._ArrayCls, attrname)
 
Ref = _RefBuilder("Ref", Array, """
A reference to a value of type `T`.
(*This is an alias for `Array[T, 0]`*)

:usage: Ref[T]

>>> x : Ref[int] = ref(0)  # Create a Ref[int]
>>> x[...] = 2             # Set it's value to 2

This base `Ref` class allows reading, but provide no guarentees about the mutability of the array via some other reference.

:meth:`Array.__getitem__` lifts side-effect-free methods and operators from `T` to `Ref`.
""")

MutableRef = _RefBuilder("MutableRef", MutableArray, """
A mutable reference to a value of type `T`.
(*This is an alias for `MutableArray[T, 0]`*)

:usage: MutableRef[T]

>>> x = ref(0)      # Create a Ref[int]
>>> x[...] = 2         # Set it's value to 2
>>> print(deref(x)) # Explicitly extract the value to pass to a none parla function.

This base `Ref` class allows reading, but provide no guarentees about the mutability of the array via some other reference.

:meth:`MutableArray.__getitem__` lifts all methods and operators from `T` to `MutableRef`.
""")

ImmutableRef = _RefBuilder("ImmutableRef", ImmutableArray, """
An immutable reference to a value of type `T`.
(*This is an alias for `ImmutableArray[T, 0]`*)

:usage: ImmutableRef[T]
""")
    

def freeze(a : Array[T, k]) -> ImmutableArray[T, k]:
    """
    Create an immutable copy of an `Array` or `Ref`.

    :param a: An array.
    :return: An immutable copy of `a`
    """
    raise NotImplementedError()

def UNSAFE_freeze(a : Array[T, k]) -> ImmutableArray[T, k]:
    """
    Create an **unsafe** immutable `Array` or `Ref`.
    Reads from the immutable reference may be arbitrarily and inconsistently stale (because *any* caching is allowed).
    No guarentees are made even for bytes or bits within the same element; they may be from different writes to that element.

    :param a: An array.
    :return: An *unsafe* immutable reference to `a` *without copying*.
    """
    raise NotImplementedError()

def deref(a : Array[T, k], *indicies) -> T:
    """
    Get a value from an array by value instead of as a `Ref[T]`.

    :param a: An array.
    :param \*indicies: `k` indicies into the array. This may not include slices.

    :return: The value at position `indicies`.
    """
    raise NotImplementedError()

def ref(v : T) -> MutableRef[T]:
    """
    Allocate a new Ref[T] (0-d array).

    :param v: The initial value.

    :return: A mutable cell initialized to `v`.
    """
    raise NotImplementedError()

def filled(v : T, *sizes) -> MutableArray[T, len(sizes)]:
    """
    Allocate a new `len(sizes)`-d array with elements of type `T`.

    :param v: The initial value for all elements.
    :param \*sizes: The shape of the filled array.

    :return: A mutable array with shape `sizes` filled with `v`.
    """
    raise NotImplementedError()


class _zerosBuilder(GenericFunction):
    def __init__(self):
        self.__doc__ = """
        Allocate an array with element of type T and fill with the default "zero" value of this type.
        Not all types have a "zero" and if T does not have a zero then this function will raise an exception.
        `zeros` requires a type parameters, so it must always be called as `zeros[T](*sizes)`.

        :usage: zeros[T](\*sizes)

        >>> zeros[int](5, 5, 5)
        
        :param T: The element type of the array.
        :param \*sizes: The shape of the resulting array.

        :return: A mutable array with shape `sizes` filled with "zero".
        """
    def __getitem__(self, T):
        def zeros(*sizes) -> MutableArray[T, len(sizes)]:
            raise NotImplementedError()
        zeros.__doc__ = self.__doc__
        return zeros
    def __call__(self, *args):
        raise TypeError("{} is not callable without a type parameter.".format(self.__name__))
    def __getattr__(self, name):
        return getattr(self[T], name)

zeros = _zerosBuilder()

