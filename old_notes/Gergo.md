# Gergo

Gergo is a intermediate representation (IR) for compiling Parla.
It is not the only IR; it sits between Parla and LLVM (or similar backend languages).
Gergo is a "core" language in the sense that it expresses the concepts of Parla as simply as possible, with as few primitives as possible.
This makes it appropriate for high-level optimizations based on high-level properties of the program.


## Core Primitives

**TODO: Add an import statement for foreign functions which allows annotations specifying how to call should be made and what it modifies.**

```python
foreign f(e_0, ..., e_n)
```
(expression)
Call a C (or Python, C++, etc.) function with the given arguments.
This is the only way to perform actual computation.
The exact calling conventions will depend on the target language.
In general, scalar values will be converted to a C-like form and `Array`s will be passed as pointer along with compiler generated (synthetic) parameters to specify the shape and strides.

```python
x
```
(expression)
A variable reference.

```python
x = e
```
(statement)
Assign the value of `e` to variable `x`.
Variables are static single assignment.

**TODO: Do we need Phi nodes?**


### Arrays

```python
Array(T, n)(e_1, ..., e_n)
```
(expression; `e_i: Size -> Array(T, n)`)
Create an array of shape `(e_1, ..., e_n)` with elements of type `T`.
`T` must be a primitive type (int32/64, float32/64, etc) or a *simple* structure type (with only primitive leaf type).
Nested arrays and arrays of python objects are *not* allowed.
The number of dimensions (rank) of the array `n` is specified in the type.
`n` may be omitted and inferred from the number of arguments to the constructor.

*Note:* The Array type do not specify a storage layout.

```python
e_1[i_1, ..., i_n]
```
(expression; `e_1: Array(T), i_j: Size -> T`)
Get the value at position `i_1, ..., i_n` in the array `e_1`.
This is called an indexing expression.
An indexing expression is *simple* if all `i_j` are affine expressions of loop variables and constants.

```python
e_1[i_1, ..., i_n] = e_2
```
(statement; `e_1: Array(T, d_1, ..., d_n), i_j: Size, e_2: T`)
Set the value at position `i_1, ..., i_n` in the array `e_1` to `e_2`.

```python
e.copy()
```
(expression; `e: Array(T) -> Array(T)`)
Copy the array.

**TODO: Can Gergo do without Array views? Instead, encoding them as a struct containing the array and the indexing offsets.**


### Array Storage Annotations

The interface for arrays does not specify the storage.
However the storage can be controlled by these annotations.
They are written as methods on arrays that make copies, but as an optimization expressions which apply these functions to an array constructor directly, such as `Array(T)(10, 40).majority(1, 0).structure_of_arrays()`, are compiled to code that directly generates the final array.
Similarly, a sequence of storage annotations is collapsed together into a single copy operation.

These operations can only be used on `Array`s not `ArrayView`s

```python
e_1.majority(i_1, ..., i_n)
```
(expression; `e_1: Array(T), i_j: Dimension -> Array(T)`)
Create a copy of `e_1` such that the *storage* of the dimensions (identified by number) is ordered as specified by the arguments.
This does **not** change the order of the dimensions in indexing expressions; it only changes how the underlying data is stored.
This allows switching between row-major and column-major order.
*Example:* If `a[row, col]` is an array then `a.majority(1, 0)` is a copy which is stored (but not indexed) in `col`-major order regardless of its previous storage.

Optionally, the dimension numbers can be replaced by dimension-stride pairs.
In that case, the array is laid out with the specified ordering and the provided strides (the strides are given in numbers of elements, not bytes).
The strides provided must be larger than the size of the array in each dimension or this throws and exception.

**FIXME: The name `majority` is bad. "Dimension order"?**

```python
e_1.structure_of_arrays()
```
(expression; `e_1: Array(T) -> Array(T)`)
Create a copy of `e_1` which is stored as a structure of arrays.
Static arrays in T are treated as a single field, so their elements will be kept together in memory.

```python
e_1.array_of_structures()
```
(expression; `e_1: Array(T) -> Array(T)`)
Create a copy of `e_1` which is stored as an array of structures.


### Sizes and Indicies

```python
Size
```
(type)
The type of array indicies and sizes.
These are non-negative integers of a platform defined size.

```python
+, -, *, //
```
(operators)
Size supports addition, subtraction, multiplication, and truncating division.
All operations must have an integer-like type as the other operand.
Integer-like types include Sizes, Python ints, and `int(s)`s.
(*Note:* `//` is used instead of `/` because `//` is integer division in Python 3, whereas `/` is "true division" which may return a float even with integer arguments.)


## Sequential Control Flow

```python
def f(x_0 : T_0, ..., x_n : T_n) -> U:
    statments
```
(statement)
Declare a function `f` with arguments `x_i`, each with type `T_i`.
The function returns type `U`.
Types are required.
Other Python features, like default values and varadic functions, are not supported.

```python
f(e_0, ..., e_n)
```
(expression)
Call a function declared in Parla.

```python
if e:
    statements_1
else:
    statements_2
```
(statement)
Conditional with predicate `e`.

```python
if e:
    e_1
else:
    e_2
```
(expression)
Conditional *expression* with predicate `e`.

```python
for x in range(e_1, e_2, e_3):
    statements
```
(statement)
A sequential numerical for-loop over a range.

```python
while e:
    statements
```
(statement)
A sequential while loop with predicate `e`.


## Parallel Control Flow

(*Note:* All the looping constructs are single dimensional. For multi-dimensional operations nest loops.)

```python
for x in parrange(e_1, e_2, e_3):
    statements_1
```
(statement)
A parallel for-loop over a range.
The arguments are start, stop, and step in that order.
All iterations execute in parallel, and are all completed when this statement completes.

```python
reduce(parrange(e_1, e_2, e_3), e_e, e_r)
```
(expression)
Reduce in parallel over the given range.
The extractor `e_e` is called with an index and must be pure.
The reduction operation `e_r` is called with two values and must be associative and pure.
Both `e_e` and `e_r` must be function names.

Associativity of the reduction operation allows any arbitrary reduction tree to be used for the computation.
If `e_r` is annotated as commutativity `reduce` can perform reductions on any available values without concern for which index or indicies they represent.

```python
with spawn():
    statements
```
Execute `statements` as a new task.
The task will execute in parallel with any following code.

```python
with finish():
    statements
```
Execute `statements` normally and then perform a limited barrier applying to all tasks created in `statements` (statically scoped).
This block has the same semantics as the implicit barrier on a function in Cilk.


### Parallelism Restriction Annotations

By default, parallel loops execute iterations concurrently (in an undefined order and potentially in parallel).
To restrict this parallelism, Parla provides read and write annotations from which dependencies can be generated.
Since all the functions used in reduction are side-effect free, these annotations will never affect reductions.

```python
with reads(e):
    statements
```
(statement)
Tell the compiler that `statements` must read the value of `e` as if all statically enclosing loops were executed sequentially.
`e` must be a variable reference or a simple indexing expression into an array or static array view.
A static array view is one which is defined in the current function using only constants.

```python
with writes(e):
    statements
```
(statement)
Tell the compiler that `statements` writes the value `e` and these writes must occur as if all statically enclosing loops were sequential.
The same restrictions as for `reads` apply to `writes`.


```python
with lock(e):
    statements
```
(statement; `e : Lock`)
Execute `statements` with the `Lock` `e` held on the current device.
Lock state is not shared across devices so locks only provide mutual exclusion within devices.


## Function Annotations

(*Note:* Annotations can be arbitrarily combined on any function.)

```python
@variant(D)
def f():
    statements
```
(statement)
The implementation of `f` for device `D`.
This will often by something like `HOST` or `GPU`, but is extensible.
Multiple devices can be specified as separate parameters: `@variant(CPU, FPGA)`.
Specifying multiple devices means that the same implementation should be used on multiple devices, but not all.
A group of variants of a function may not include multiple implementations for the same device.
A group may contain *one* implementation without any devices and this variant will be used for any devices which do not have specific implementations in the group.

```python
@pure
def f():
    statements
```
(statement)
Declare the function `f` as *pure*.
Where pure means:
* Its return value depends only on its parameters and not on any internal or external state; and
* Its evaluation has no side effects.

```python
@associative
def f(a, b):
    statements
```
(statement)
Declare the binary function `f` as associative.

```python
@commutative
def f(a, b):
    statements
```
(statement)
Declare the binary function `f` as commutative.


## Types

All Gergo types are pass-by-value.
However Gergo provides a `Ref` type constructor which is a value type that forwards to referenced value, providing explicitly reference semantics.
Traditional pass-by-reference arrays have type `Ref[Array[...]]`.

Up-casts (to supertypes) are automatic.
Down-casts (to subtypes) are not supported.
Supporting downcasts would either require runtime type information and create the potential for runtime type errors, or would create complete memory unsafety.

The types defined in [Types.md](Types.md) are modified slightly for Gergo.
Specifically, indexing expressions may not contain an ellipsis (so all dimensions must be explicitly handled).
**TODO: There are probably more simplifications.**


## Syntax

Defined the metavariables used above either bare or subscripted.

```python
statements
```
Some number of program statements in Python syntax.
It is indentation delimited and must contain at least one statement.
Execution is sequential.

```python
e
```
An expression in Gergo.
Generally Python like, but more restricted.

```python
s
```
A slicing expression.

```python
x, y
```
A variable name.

```python
T, U
```
A type expression.


