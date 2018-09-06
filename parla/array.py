"""
Parla provides multi-dimensional arrays similar to :class:`numpy.ndarray` in :std:doc:`Numpy <numpy:docs/index>`.

Indexing
--------

Indexing in Parla is similar to :std:doc:`Numpy <numpy:docs/index>`.
However, Parla eliminates many of the special cases and simplifies whenever possible.

.. todo:: Fully explain indexing expressions. The documentation here should be cleanly combined with that for `Array.__getitem__`.

.. testsetup::

   from __future__ import annotations
   from parla.array import *

"""

# """
# Arrays support a range on indexing and slicing operations.
# These operations take an indexing expression to select a subset of the array to operate on or return.
# For an `Array[T, k] <Array>`, the indexing expression is (conceptually) `k` slices or indicies: `i₁, …, iₖ`.
# Each `iₓ` specifies the selected elements in dimension `x` as either an index of a single element or a slice expression (define below).
# A literal `...` (ellipsis) represents enough slices to provide `k` slices or indicies to the array.
# The represented slices select the entire dimension.
# For example, if `a : Array[T, 3]` then in `a[1, 2:, ...]` the ellipsis represents `:, :, :` (where `:` is the universal slice).
# Only one ellipsis can appears in an indexing expression, but it can appear anywhere.
# If there are fewer indicies and slices than dimensions an ellipsis is required to explicitly handle the remaining dimensions.

# .. todo:: Fix up indexing documentation and cover newdim. The documentation here should be cleanly combined with that for `Array.__getitem__`.


# The type of the expected or returned array depends on the how many slices are used in indexing.
# If there are `p` slices (and `k - p` indicies) used then the expected or returned array has type `Array[T, p] <Array>`, and the remaining `p` dimensions are those which were sliced instead of indexed.
# If `p == 0` then the return type is `Array[T, 0] ≡ Ref[T]`.
# This means that the returned types (and values) of `a[2]` and `a[2:3]` are different despite the results selecting the same elements from `a`; 
# `a[2]` eliminates the dimension and `a[2:3]` keeps it (with length 1).


# Slice expressions are written `start:end:step` where `start`, `end` are indicies and `step` is a integer stride.
# The second colon can be omitted, `start:end`, for a `step` of 1.
# The `start` and `end` values can be omitted for 0 and the last element respectively.
# Slices have the same semantics as Python slices: `i:j:k` selects all elements of an array with an index `x` where `x = i + n*k`, `n >= 0` and `i <= x < j` (adapted from the `Python documentation <https://docs.python.org/3.7/reference/datamodel.html#types>`_).
# """

from __future__ import annotations

from parla.typing import *
from parla import function_decorators, detail

__all__ = [
    "Array", "ImmutableArray", "MutableArray",
    #"InplaceArray", "ImmutableInplaceArray", "MutableInplaceArray",
    #"Ref", "ImmutableRef", "MutableRef", "ref",
    "zero", "full",
    "newdim", "infer",
    "inplace", "layout",
]

T = TypeVar("T")
k = LiftedNaturalVar("k")
s1 = LiftedNaturalVar("s₁")
sk = LiftedNaturalVar("sₖ")

class inplace(detail.Detail):
    """
    An `Array` type detail which specifies that the array will always have the specified shape exactly.
    This allow the whole array to be stored in a fixed size region inside a `~parla.struct.Struct` and be accessed without a pointer dereference.
    """
    def __init__(self, *shape):
        """
        :param \*shape: The shape all instances will have.
        """
        self.shape = shape
        self.args = shape

class layout(detail.Detail):
    """
    A detail specifying the storage layout for arrays.
    Storage layouts are specific to an dimensionality and (sometimes) an element type.
    """
    def __init__(self, *dim_order, soa : bool = False):
        """
        :param \*dim_order: A series of dimension numbers ordered from slowest to fastest changing.
                            So, `layout(0, 1)` is row-major order and `layout(1, 0)` is column-major.
                            The dimension numbers are with respect to the indexing order of the array this layout is applied to.
                            Every dimention must be listed exactly once.
        :param soa: Use a structure of arrays layout if the element type is a `Struct` and `soa = True`.
        """
        self.dim_order = dim_order
        self.soa = soa
        self.args = dim_order + (soa,)

class _NewDim:
    """
    This sentinal value is used in place of a slice or index to create a new dimension (of length 1) during indexing.
    """

    def __str__(self):
        return "newdim"
    def __repr__(self):
        return "newdim"

newdim = _NewDim()

class _Infer:
    """
    This sentinal value specifies that the value it replaces should be inferred based on other values.

    .. seealso:: `Array.reshape`, `Array.reshape_view`
    """

    def __str__(self):
        return "infer"
    def __repr__(self):
        return "infer"

infer = _Infer()

def count_slices(indexing_expression):
    """
    Count the number of slices used in an indexing expression.
    """
    return len([s for s in indexing_expression if isinstance(s, slice)])

class Array(TypeConstructor[T, k], detail.DetailableType):
    """
    Parla arrays are (generally) references to data stored outside the array value itself.
    Arrays are parameterized on the element type `T` and the dimensionality (rank) `k` (a non-negative integer).

    :usage: Array[T, k]

    >>> x : Array[int, 4]

    This base `Array` class allows reading, but provide no guarentees about the mutability of the array via some other reference.

    :meth:`Array.__getitem__` lifts `side-effect-free <parla.function_decorators.side_effect_free>` methods and operators from `T` to `Array`.

    The number of dimensions `k` may be 0.
    A 0-d array has an empty shape `()` and exactly one element (indexed by the empty tuple of indicies).
    Parla uses 0-d arrays as references mutable scalar values.
    These 0-d arrays are returned from array indexing (for example, `zero[int](3, 3)[0, 0]` has type `Array[int, 0]`) and can be created directly using an empty shape.
    """

    __type_details__ = frozenset((inplace, layout))
    __instance_details__ = frozenset((layout,))
    
    def __getitem__(self, indexing_expression) -> Array[T, count_slices(indexing_expression)]:
        """
        >>> a = zero[int](3, 3)
        >>> a[1:, 2] : Array[int, 1]
        >>> a[1:, 2].shape
        (2,)
        >>> a[1:, :] : Array[int, 2]
        >>> a[1:, :].shape
        (2, 3)
        >>> a[1:, ...].shape
        (2, 3)
        >>> a[1, 2] : Array[int, 0]

        :param indexing_expression: An indexing expression made up of indicies and slices, and optionally an ellipsis.
            Like numpy and Python, `Array` supports negative indices to index from the end of a dimension.
            Slices with negative indicies are also supported, however a single slice must be totally negative or totally non-negative.
        :return: The view on `self` explosing the selected parts.
        :raise ValueError: if a slice in `indexing_expression` contains both negative and non-negative numbers.
        :raise IndexError: if an index (or element of a slice) is outside the dimensions of the array.
        :allocation: Never
        """
        raise NotImplementedError()

    def get(self, *indices) -> T:
        """
        >>> a = zero[int](3, 3)
        >>> a.get(1, 2) : int

        :param \*indices: A series of indices specifying a location in the array. Slices are not allowed.
        :type \*indices: exactly `k` integers
        :return: the value in a single element of this array.
        :raise IndexError: if an index (or element of a slice) is outside the dimensions of the array.
        :allocation: Never
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        Return an iterator over the outer-most (left-most) dimension, returning views.
        The resulting views are `(k - 1)`-d, as if they were created with `self[i, ...]` where `i` is the index variable for this iterator.

        :usage: `iter(self)`

        :return: A `~parla.loops.ParIterator` over the outer-most dimension of `self`.
        :allocation: Never

        .. seealso:: `~parla.loops.iter`
        """
        raise NotImplementedError()

    def __getattr__(self, attrname):
        """
        Get the lifted version of an attribute on `T`.
        All `side-effect-free <parla.function_decorators.side_effect_free>` methods and operators (including pure) on `T` are lifted to `Array[T, k]` as element-wise operations.

        :usage: `self.attrname`

        For example, given a structure `S`:

        .. testsetup::
          from parla.struct import *
          from parla.primitive import *

        >>> class S(Struct):
        ...     x: I[32] = 1
        >>> a = zero[S](2, 2)
        >>> a.x : Array[I[32], 2]
        [[1, 1], [1, 1]]

        :return: The lifted bound method or `Array` of attribute values.
        :allocation: Lifting the attribute never requires allocation, however lifted operations themselves often require an output array to be allocated.
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
        
    def permute_dims(self, *dim_order):
        """
        Create a view with rearranged the dimensions.

        >>> a = zero[int](2, 3, 4)
        >>> a.permute_dims(2, 0, 1)
        >>> a.shape
        (4, 2, 3)

        :param \*dim_order: The new order of dimensions specified as a series of current dimension numbers (0 based).

        :return: A view on this array which expects indicies in the new order.
        :allocation: Never
        """
        raise NotImplementedError()

    def reshape(self, *shape):
        """
        Reshape this array, performing a copy if needed.
        A single dimension length may be `infer` to compute that length based on the other dimensions and the total number of elements in `self`.

        >>> a.shape(infer)        # Flatten the array to 1-d
        >>> a.shape(3, 3)         # Reshape to 3 by 3 if possible
        >>> a.shape(3, infer)     # Reshape to 3 by x, where x is number of elements in `self` divided by 3 (if divisible)

        If a the reshaped version is required, write `a.reshape(shape).copy()`.
        The compiler will eliminate the potential additional copy.

        :raise ValueError: if the new shape has a different number of elements than `self`.
        :allocation: If the new shape is incompatible with this array's storage layout.

        .. seealso:: `reshape_view`
        """
        try:
            return self.reshape_view(shape)
        except ValueError:
            raise NotImplementedError()

    def reshape_view(self, *shape):
        """
        Create a view of this array with a new shape.
        Element ordering is defined by the view of `self`.
        `reshape_view` never performs a copy.

        Not all arrays support views of all shapes.
        `reshape_copy` supports all shapes on all arrays, but requires a copy.

        :raise ValueError: if the reshaped view cannot be created due to the storage layout of `self`.
        :allocation: Never

        .. seealso:: `reshape`
        """
        raise NotImplementedError()


    def copy(self, *, requirements: tuple = (), hints: tuple = ()):
        """
        Copy this array.
        The copy is will never fail due to `layout` requirements.

        :param requirements: Any additional requirements to apply to the returned array.
        :param hints: Any additional hints to apply to the returned array.
        :return: A new array with the same type and shape as self.
        :raise ValueError: if `layout` is not applicable to this array's dimensionality and element type.
        :allocation: Always
        """
        raise NotImplementedError()

    @property
    def shape(self):
        """
        The shape of `self` represented as a tuple of dimension lengths.
        :allocation: Never
        """
        raise NotImplementedError()

    def freeze(self) -> ImmutableArray[T, k]:
        """
        Create an immutable copy of an `Array`.

        >>> a = zero[F[32]](3, 3)
        >>> a[0, 0] = 1
        >>> a[1, 1] = 1
        >>> b = a.freeze()   # Create an ImmutableArray copy
        >>> a[2, 2] = 1      # Writing to a does not affect b

        The optimizer will attempt to optimize freeze by replacing ``a = a.freeze()`` with ``a = a.UNSAFE_freeze()`` (see `UNSAFE_freeze`) when the original value of `a` does not escape (making the optimization safe).

        :return: An immutable copy of this array.
        :rtype self.Immutable:
        :allocation: Usually (optimization may eliminate it)
        """
        raise NotImplementedError()

    def UNSAFE_freeze(self) -> ImmutableArray[T, k]:
        """
        Create an **unsafe** immutable `Array`.
        Reads from the immutable reference may be arbitrarily and inconsistently stale (because *any* caching is allowed).
        No guarentees are made even for bytes or bits within the same element; they may be from different writes to that element.

        >>> a = zero[F[32]](3, 3)
        >>> a[0, 0] = 1
        >>> a[1, 1] = 1
        >>> b = a.UNSAFE_freeze() # Create an ImmutableArray *reference* to a
        >>> # a[2, 2] = 1         # This would make b[2, 2] undefined
        >>> a = b                 # Better to destroy the reference to a

        :return: An *unsafe* immutable reference to this array *without copying*.
        :allocation: Never

        .. seealso:: `freeze`
        """
        raise NotImplementedError()

class MutableArray(Array):
    """
    A reference to a mutable `Array`.

    :usage: MutableArray[T, k]

    :meth:`MutableArray.__getitem__` lifts all methods and operators from `T` to `Array`.
    """
    def __getitem__(self, indexing_expression) -> MutableArray[T, count_slices(indexing_expression)]:
        return super().__getitem__(indexing_expression)

    def __setitem__(self, indexing_expression, a : Array[T, count_slices(indexing_expression)]):
        """
        Assign new values to the slice.

        :param indexing_expression: An indexing expression made up of indices and slices, and optionally an ellipsis. See `__getitem__` for details.
        :param a: The array to copy into the slice. `a` will be broadcast as needed.
        :allocation: Never
        """
        raise NotImplementedError()

    def __getattr__(self, attrname):
        """
        Get the lifted version of an attribute on `T`.
        *All* methods and operators (including side-effecting) on `T` are lifted to `Array[T, k]` as element-wise operations.

        In-place operators (such as `+=`) are supported when the left operand is of the same type as the return value (that is, `(x : T) + (y : U)` has type `T`).
        Parla guarantees this for all :mod:`primitive types<primitives>`.
        Unsupported in-place operators will raise a TypeError.

        :return: The lifted bound method or `Array` of attribute values.
        :allocation: Lifting the attribute never requires allocation, however lifted operations themselves often require an output array to be allocated. In-place operators never allocate.
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

# class InplaceArray(Array[T, s1, ..., sk], TypeConstructor[T, k]):
#     """
#     Parla in-place arrays have a fixed shape and store their data along with their metainformation (instead of storing a reference to external data).
#     This means that in-place arrays do not result in pointers and are similar to structures with numbered fields, all of the same type.
#     `InplaceArray` is parameterized on the element type `T` and the shape `s₁, …, sₖ`.

#     :usage: InplaceArray[T, s₁, …, sₖ]

#     >>> x : InplaceArray[float, 4, 4]

#     Here `x` is a 4 by 4 in-place array of floats which could store a 3-d coordinate transform.

#     `InplaceArray` exposes the same API as `Array` and views on `InplaceArray` are `Array` (or the appropriate mutability subtype).
#     If `T` has a "zero" then the "zero" of `InplaceArray[T, s₁, …, sₖ]` is the inplace array filled with `T`'s "zero".

#     .. todo:: How are InplaceArrays created?
#     """
#     pass

# class MutableInplaceArray(MutableArray, InplaceArray):
#     """
#     A mutable in-place array.

#     :usage: MutableInplaceArray[T, s₁, …, sₖ]

#     :see: `InplaceArray`
#     """
#     pass

# class ImmutableInplaceArray(ImmutableArray, InplaceArray):
#     """
#     An immutable in-place array.

#     :usage: ImmutableInplaceArray[T, s₁, …, sₖ]

#     :see: `InplaceArray`
#     """
#     pass


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
(This is an alias for :class:`Array[T, 0] <Array>`)

:usage: Ref[T]

>>> x : Ref[int] = ref(0)  # Create a Ref[int]
>>> print(deref(x))        # Explicitly extract the value to pass to a non-Parla function.

This base `Ref` class allows reading, but provide no guarentees about the mutability of the array via some other reference.

:meth:`Array.__getitem__` lifts side-effect-free methods and operators from `T` to `Ref`.
""")

MutableRef = _RefBuilder("MutableRef", MutableArray, """
A mutable reference to a value of type `T`.
(This is an alias for :class:`MutableArray[T, 0] <MutableArray>`)

:usage: MutableRef[T]

>>> x = ref(0)      # Create a Ref[int]
>>> x[...] = 2      # Set it's value to 2

:meth:`MutableArray.__getitem__` lifts all methods and operators from `T` to `MutableRef`.
""")

ImmutableRef = _RefBuilder("ImmutableRef", ImmutableArray, """
An immutable reference to a value of type `T`.
(This is an alias for :meth:`ImmutableArray[T, 0] <ImmutableArray>`)

:usage: ImmutableRef[T]
""")


## Functions

def ref(v : T) -> MutableRef[T]:
    """
    Allocate a new Ref[T] (0-d array).

    :param v: The initial value.

    :return: A mutable cell initialized to `v`.
    """
    raise NotImplementedError()

def full(v : T, *shape) -> MutableArray[T, len(shape)]:
    """
    Allocate a new `len(shape)`-d array with elements of type `T`.

    :param v: The initial value for all elements.
    :param \*shape: The shape of the array.

    :return: A mutable array with shape `shape` filled with `v`.
    """
    raise NotImplementedError()


class _zeroBuilder(GenericFunction):
    def __init__(self):
        self.__doc__ = """
        Allocate an array with element of type T and fill with the default "zero" value of this type.
        Not all types have a "zero" and if T does not have a zero then this function will raise an exception.
        The type parameter `T` is required (since it cannot be inferred from any other argument).
        To avoid an explicit type use :meth:`filled`.

        :usage: zero[T](\*shape)

        >>> a : Array[int, 3] = zero[int](5, 5, 5)
        >>> a.shape
        (5, 5, 5)

        :param T: The element type of the array.
        :param \*shape: The shape of the resulting array.

        :return: A mutable array with shape `shape` filled with "zero".
        """
    def __getitem__(self, T):
        def zero(*shape) -> MutableArray[T, len(shape)]:
            raise NotImplementedError()
        zero.__doc__ = self.__doc__
        return zero
    def __call__(self, *args):
        raise TypeError("{} is not callable without a type parameter.".format(self.__name__))
    def __getattr__(self, name):
        return getattr(self[T], name)

zero = _zeroBuilder()
