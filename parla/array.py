"""
Arrays support a range on indexing and slicing operations.
These operations take an indexing expression to select a subset of the array to operate on or return.
For an `Array[T, k] <Array>`, the indexing expression is (conceptually) `k` slices or indicies: `i₁, …, iₖ`.
Each `iₓ` specifies the selected elements in dimension `x` as either an index of a single element or a slice expression (define below).
A literal `...` (ellipsis) represents enough slices to provide `k` slices or indicies to the array.
The represented slices select the entire dimension.
For example, if `a : Array[T, 3]` then in `a[1, 2:, ...]` the ellipsis represents `:, :, :` (where `:` is the universal slice).
Only one ellipsis can appears in an indexing expression, but it can appear anywhere.
If there are fewer indicies and slices than dimensions an ellipsis is required to explicitly handle the remaining dimensions.


The type of the expected or returned array depends on the how many slices are used in indexing.
If there are `p` slices (and `k - p` indicies) used then the expected or returned array has type `Array[T, p] <Array>`, and the remaining `p` dimensions are those which were sliced instead of indexed.
If `p == 0` then the return type is `Array[T, 0] ≡ Ref[T]`.
This means that the returned types (and values) of `a[2]` and `a[2:3]` are different despite the results selecting the same elements from `a`; `a[2]` eliminates the only dimension and `a[2:3]` keeps that dimension with length 1.


Slice expressions are written `start:end:step` where `start`, `end` are indicies and `step` is a integer stride.
The second colon can be omitted, `start:end`, for a `step` of 1.
The `start` and `end` values can be omitted for 0 and the last element respectively.
Slices have the same semantics as Python slices: `i:j:k` selects all elements of an array with an index `x` where `x = i + n*k`, `n >= 0` and `i <= x < j` (adapted from the Python documentation: https://docs.python.org/3.7/reference/datamodel.html#types).
"""

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
        The type parameter `T` is required (since it cannot be inferred from any other argument).
        To avoid an explicit type use :meth:`filled`.

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

