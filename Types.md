# Type system for Parla

This describes a type system for values in Parla (and Gergo).

## Types

Parla's types fall into 2 kinds: Sized and Unsized.
Sized types have a statically known size (known at compile time).
Unsized types have a dynamic size (known only at run time), so they cannot be placed directly in arrays or structures.
Parla types do not specify a storage layout or implementation.
A type specifies an interface to values of that type and requires the implementation to provide type meta-information at compile time.
All types are immutable except for those marked explicitly as mutable.

### Unsized Types
**Arrays:**
Parla arrays are parameterized on the *sized* type `T` and the dimensionality (rank) `k` (a positive integer).
```
    ● Array[T, k]               (Allows reading)
    ├─● Array[T, k].Mutable     (Allows writing)
    └─● Array[T, k].Immutable   (No one can write this)
```

### Sized Types

Parla provides reference to objects stored elsewhere.
`Ref` is parameterized on a type `T` which need not be sized.
```
    ● Ref[T]     (A reference to a value of type T)
```

Parla also provides a number of primitive types.
```
    ● Boolean    (Boolean)
    ● I[size]    (Signed Integer)
    ● UI[size]   (Unsigned Integer)
    ● F[size]    (IEEE-like Floating-point)
```
The `size` parameter for the numeric types specifies the size of the value in *bits*.
Implementations will probably restrict `size` to a set, such as {8, 16, 32, 64, 128} for `I` and `UI`, and {16, 32, 64} for `F`.



**Structures:**
For each user-defined structure `S`, Parla generates three types.
```
    ● S
    ├─● S.Mutable
    └─● S.Immutable
```

**Sized Arrays:**
Sized arrays are tree of subtype of the `Array` tree above which include the array bounds (shape) in addition to the element type.
```
    ┌─ Array[T, k] ┄
    ● SizedArray[T, s₁, …, sₖ]
    │ ┌─ Array[T, k].Mutable ┄
    ├─● SizedArray[T, s₁, …, sₖ].Mutable
    │ ┌─ Array[T, k].Immutable ┄
    └─● SizedArray[T, s₁, …, sₖ].Immutable
```

#### Atomic Types

This section describes some types that we could support if needed, but are not required for straight-forward parallel programming in Parla.

These types provide atomicity guarantees only within a single device.
If multiple devices share access to a single memory, then atomicity is only guaranteed if operations only happen from one of those devices.
If multiple devices perform operations, the state of these values become undefined and operations on them are undefined (within the scope of the value).

```
    ● Lock       (A lock or mutex)
```

**Atomic primitive types:**
Primitive-like types which provide atomic update and access semantics.
```
    ● AtomicBoolean    (Atomic Boolean)
    ● AtomicI[size]    (Atomic Signed Integer)
    ● AtomicUI[size]   (Atomic Unsigned Integer)
    ● AtomicF[size]    (Atomic IEEE-like Floating-point)
```

## Programming Interfaces

Parlas types each provide an interface of operators and methods (which are not different other than syntax).
Here we will describe the interfaces for the types above.

### Array Types

Arrays support a range on indexing and slicing operations.
These operations take an indexing expression to select a subset of the array to operate on or return.
For an `Array[T, k]`, the indexing expression is (conceptually) `k` slices or indicies: `i₁, …, iₖ`.
Each `iₓ` specifies the selected elements in dimension `x` as either an index of a single element or a slice expression (define below).
A literal `...` (ellipsis) represents enough slices to provide `k` slices or indicies to the array.
The represented slices select the entire dimension.
For example, if `a : Array[T, 3]` then in `a[1, 2:, ...]` the ellipsis represents `:, :, :` (where `:` is the universal slice).
Only one ellipsis can appears in an indexing expression, but it can appear anywhere.
If there are fewer indicies and slices than dimensions an ellipsis is required to explicitly handle the remaining dimensions.


The type of the expected or returned array depends on the how many slices are used in indexing.
If there are `p > 0` slices (and `k - p` indicies) used then the expected or returned array has type `Array[T, p]`, and the remaining `p` dimensions are those which were sliced instead of indexed.
If there are no slices, then the returned or expected type is `T`.
This means that the returned types (and values) of `a[2]` and `a[2:3]` are different despite the results selecting the same elements from `a`; `a[2]` eliminates the only dimension and `a[2:3]` keeps that dimension with length 1.


Slice expressions are written `start:end:step` where `start`, `end` are indicies and `step` is a integer stride.
The second colon can be omitted, `start:end`, for a `step` of 1.
The `start` and `end` values can be omitted for 0 and the last element respectively.
Slices have the same semantics as Python slices: `i:j:k` selects all elements of an array with an index `x` where `x = i + n*k`, `n >= 0` and `i <= x < j` (adapted from the [Python documentation](https://docs.python.org/3.7/reference/datamodel.html#types)).


```python
self : Array[T, k]
    self[indexing_expression] -> Array[T, m] or T            # (Slice or indexed read)
    (for example,
      self[i₁ : UI, …, iₖ : UI] -> T
      self[i : UI, ...] -> Array[T, k - 1]
      self[i₁ : Slice, …, iₖ : Slice] -> Array[T, k]
    )

  if T is a structure S with fields (f₁ : U₁, …, fₘ : Uₘ):
    self.fₓ -> Array[Uₓ, m]                                  # (Field slice read)

  if T is a numeric type:
    self + v : T -> Array[T, m]                              # (Copying scalar addition)
    similar operators for -, *, /, **
    
    self + a : Array[T, m] -> Array[T, m]                    # (Copying element-wise addition)
    similar operators for -, *, /, ** (all element-wise)


self : Array[T, k].Mutable (extends Array[T, k])
    self[indexing_expression] = x : Array[T, m] or T         # (Slice assignment or indexed write)
    (for example,
      self[i₁ : UI, …, iₖ : UI] = x : T
      self[i : UI, ...] = x : Array[T, k - 1]
      self[i₁ : Slice, …, iₖ : Slice] = x : Array[T, k]
    )

    self[indexing_expression] -> Array.Mutable[T, m] or T    # (Mutable slice or indexed read)
    
    self.freeze()                                            # (Frozen copy)

  if T is a structure S with fields (f₁ : U₁, …, fₘ : Uₘ):
    self.fₓ = v : Array[Uₓ, m]                               # (Field slice write)

  if T is a numeric type:
    self += v : T                                            # (In-place scalar addition)
    similar operators for -, *, /, **
    
    self + a : Array[T, m]                                   # (In-place element-wise addition)
    similar operators for -, *, /, ** (all element-wise)


self : Array[T, k].Immutable (extends Array[T, k])
    self[indexing_expression] -> Array.Immutable[T, m] or T  # (Immutable slice or indexed read)
```

Additional element-wise operators can be supported as needed.

**TODO: How do we support broadcasting? (I think it should be explicit)**

`SizedArray`s provide no additional operations over their corresponding `Array` types.
They only provide additional static information to the compiler which enables them to be array and structure elements and potentially allows the compiler to flag errors.

#### Escape Hatch Operations

`Array[T, k]` also provides an unsafe casting operations `self.unsafe_freeze() : Array[T, k].Immutable` which returns `self` with type `Array[T, k].Immutable` *without copying*.
If the program ever modifies `self` after calling this, the contents of the immutable reference are undefined.


### Ref Type

The type `Ref[T]` is an immutable reference to an underlying value (which may be mutable).
`Ref[T]` values have exactly the operations of `T`.

### Structure Types

Structure types are defined as:
```
struct S:
    f₁ : T₁
    …
    fₖ : Tₖ
```
where each `fₓ` is a field name (an identifier) and each `Tₓ` is a sized type.
For each structure `S`, Parla provides three types whose interfaces are:
```python
self : S
    self.fₓ : Tₓ                  # (Field selection)

self : S.Mutable (extends S)
    self.fₓ = v : Tₓ              # (Field write)
    self[...] = v : S             # (All write)
    self.freeze() : S.Immutable   # (Frozen copy)

self : S.Immutable (extends S)
    # (no additional operations)
```

#### Escape Hatch Operations

`S` also provides an unsafe casting operations `self.unsafe_freeze() : S.Immutable` which returns `self` with type `S.Immutable` *without copying*.
If the program ever modifies `self` after calling this the values of the fields of the immutable reference are undefined.

### Lock Type

`Lock` is a sized type which provides a mutex interface.
```python
self : Lock
    self.lock()                   # (Acquire)
    self.unlock()                 # (Release)
    self.try_lock() : Boolean     # (Non-blocking acquire)
    with self: …                  # (Synchronized block on self)
```

### Atomic Primitive Types

The atomic primitive types implement interfaces which are different from their non-atomic counter parts.
The operations are a subset of the simple [atomic operations in Java](https://docs.oracle.com/javase/10/docs/api/java/util/concurrent/atomic/AtomicInteger.html).
The operations provide several different operations for ordering semantics, and basic atomic read-update-write operations (such as, compare-and-swap and increment-and-get).

**TODO: Specify the exact operations if we need this.**
