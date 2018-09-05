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

.. todo:: Fix up indexing documentation and cover newdim. The documentation here should be cleanly combined with that for `Array.__getitem__`.


The type of the expected or returned array depends on the how many slices are used in indexing.
If there are `p` slices (and `k - p` indicies) used then the expected or returned array has type `Array[T, p] <Array>`, and the remaining `p` dimensions are those which were sliced instead of indexed.
If `p == 0` then the return type is `Array[T, 0] ≡ Ref[T]`.
This means that the returned types (and values) of `a[2]` and `a[2:3]` are different despite the results selecting the same elements from `a`; `a[2]` eliminates the dimension and `a[2:3]` keeps it (with length 1).


Slice expressions are written `start:end:step` where `start`, `end` are indicies and `step` is a integer stride.
The second colon can be omitted, `start:end`, for a `step` of 1.
The `start` and `end` values can be omitted for 0 and the last element respectively.
Slices have the same semantics as Python slices: `i:j:k` selects all elements of an array with an index `x` where `x = i + n*k`, `n >= 0` and `i <= x < j` (adapted from the `Python documentation <https://docs.python.org/3.7/reference/datamodel.html#types>`_).
"""

from __future__ import annotations

from parla.typing import *
from parla import function_decorators

__all__ = [
    "Array", "ImmutableArray", "MutableArray",
    "InplaceArray", "ImmutableInplaceArray", "MutableInplaceArray",
    "Ref", "ImmutableRef", "MutableRef",
    "ref", "zeros", "filled",
    "newdim"
]

T = TypeVar("T")
k = LiftedNaturalVar("k")
s1 = LiftedNaturalVar("s₁")
sk = LiftedNaturalVar("sₖ")

class _NewDim:
    """
    This sentinal value is used in place of a slice or index to create a new dimension (of size 1) during indexing.
    """

    def __str__(self):
        return "newdim"
    def __repr__(self):
        return "newdim"

newdim = _NewDim()

class StorageLayout:
    """
    A representation of a storage layout for arrays.
    Storage layouts are specific to an dimensionality and (sometimes) an element type.
    """
    
    def __init__(self, *dim_order, soa : bool = False):
        """
        :param \*dim_order: A series of dimension numbers ordered from slowest to fastest changing.
                            So, `StorageLayout(0, 1)` is row-major order and `StorageLayout(1, 0)` is column-major.
                            The dimension numbers are with respect to the indexing order of the array this layout is applied to.
        :param soa: Use a structure of arrays layout if the element type is a `Struct` and `soa = True`.
        """
        self.dim_order = dim_order
        self.soa = soa

def count_slices(indexing_expression):
    """
    Count the number of slices used in an indexing expression.
    """
    return len([s for s in indexing_expression if isinstance(s, slice)])

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
        >>> a = zeros[int](3, 3)
        >>> a[1:, 2] : Array[int, 1]
        >>> a[1:, 2].shape
        (2,)
        >>> a[1:, :] : Array[int, 2]
        >>> a[1:, :].shape
        (2, 3)
        >>> a[1:, ...].shape
        (2, 3)
        >>> a[1, 2] : Ref[int]

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
        >>> a = zeros[int](3, 3)
        >>> a.get(1, 2) : int

        :param \*indices: A series of indices specifying a location in the array. Slices are not allowed.
        :type \*indices: exactly `k` integers
        :return: the value in a single element of this array.
        :raise IndexError: if an index (or element of a slice) is outside the dimensions of the array.
        :allocation: Never
        """
        raise NotImplementedError()

    def permute_dims(self, *dim_order):
        """
        Create a view with rearranged the dimensions.

        >>> a = zeros[int](2, 3, 4)
        >>> a.permute_dims(2, 0, 1)
        >>> a.shape
        (4, 2, 3)

        :param \*dim_order: The new order of dimensions specified as a series of current dimension numbers (0 based).

        :return: A view on this array which expects indicies in the new order.
        :allocation: Never
        """
        raise NotImplementedError()

    def __getattr__(self, attrname):
        """
        Get the lifted version of an attribute on `T`.
        All `side-effect-free <parla.function_decorators.side_effect_free>` methods and operators (including pure) on `T` are lifted to `Array[T, k]` as element-wise operations.

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

    def reshape(self, shape):
        """
        Reshape this array, performing a copy if needed.

        If a the reshaped version is required, write `a.reshape(shape).copy()`.
        The compiler will eliminate the potential additional copy.

        :see: `reshape_view`
        :allocation: If the new shape is incompatible with this array's storage layout.
        """
        try:
            return self.reshape_view(shape)
        except ValueError:
            raise NotImplementedError()

    def reshape_view(self, shape):
        """
        Create a view of this array with a new shape.
        Element ordering is defined by the view of `self`.
        `reshape_view` never performs a copy.

        Not all arrays support views of all shapes.
        `reshape_copy` supports all shapes on all arrays, but requires a copy.

        :raise ValueError: if the reshaped view cannot be created due to the storage layout of `self`.
        :allocation: Never
        """
        raise NotImplementedError()


    def copy(self, *, layout: StorageLayout = None, **hints):
        """
        Copy this array.
        The copy is guaranteed to have the requested `storage` layout (see the :meth:`hint() argument storage<hint>`) if it is provided to this call.

        :param StorageLayout layout: The required storage layout. `None` means the compiler will choose a layout.
        :param hints: Any additional hints to apply to the returned array.
        :return: A new array with the same type and shape as self.
        :raise ValueError: if `layout` is not applicable to this array's dimensionality and element type.
        :allocation: Always
        """
        raise NotImplementedError()

    def hint(self, *, layout: StorageLayout = None):
        """
        Provide compilation hints and requests to the compiler.
        The compiler will produce (optional) warnings if the hints are not followed.

        :param StorageLayout layout: Request that the array have the storage layout described.
                        This does *not* change the order of the dimensions in indexing expressions or how fields are accessed; it only changes how the underlying data is stored in memory.

        :return: A hinted iterator based on `self`.
        :raise ValueError: if `layout` is not applicable to this array's dimensionality and element type.
        :allocation: Never
        """
        return self

    @property
    def shape(self):
        """
        The shape of `self` represented as a tuple of dimension lengths.
        """
        raise NotImplementedError()

    def freeze(self) -> ImmutableArray[T, k]:
        """
        Create an immutable copy of an `Array` or `Ref`.

        :return: An immutable copy of this array.
        """
        raise NotImplementedError()

    def UNSAFE_freeze(self) -> ImmutableArray[T, k]:
        """
        Create an **unsafe** immutable `Array` or `Ref`.
        Reads from the immutable reference may be arbitrarily and inconsistently stale (because *any* caching is allowed).
        No guarentees are made even for bytes or bits within the same element; they may be from different writes to that element.

        :return: An *unsafe* immutable reference to this array *without copying*.
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

class MutableInplaceArray(MutableArray, InplaceArray):
    """
    A mutable in-place array.

    :usage: MutableInplaceArray[T, s₁, …, sₖ]

    :see: `InplaceArray`
    """
    pass

class ImmutableInplaceArray(ImmutableArray, InplaceArray):
    """
    An immutable in-place array.

    :usage: ImmutableInplaceArray[T, s₁, …, sₖ]

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

        >>> a : Array[int, 3] = zeros[int](5, 5, 5)
        >>> a.shape
        (5, 5, 5)

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
