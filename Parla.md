# Parla

Parla uses a python like syntax which we will tweak to make it actually valid Python code for the prototype (and maybe the final version).


## Core Primitives


### Foreign Function Interface

To call a foreign function from within Parla code, the function needs to be imported.
The import will provide a raw callable which accepts special C based types.
(If the prototype runs in python this could be [CFFI](https://cffi.readthedocs.io/en/latest/).)
In general, the raw callable will need to be wrapped to provide an easy to use Parla interface.
This wrapper uses unsafe operations to extract the low-level pointers and structural information from arrays and potentially convert other types.
Utility methods exist for external functions which support Dynd.

The wrapper for a `gemv` would be:
```python
def sgemv(alpha : F[32], a : Array[F[32], 2], x : Array[F[32], 1], beta : F[32], y : Array[F[32], 1].Mutable):
    ???Determine if a has an appropriate layout and set Layout; copy the array if the layout is not gemv compatible.
    cblas_sgemv(???Layout, ???trans, 
        a.size(0), a.size(1), alpha, a.UNSAFE_data_address(), a.UNSAFE_stride(0), 
        x.UNSAFE_data_address(), x.UNSAFE_stride(0), 
        beta, y.UNSAFE_data_address(), y.UNSAFE_stride(0));
```
This wrapper is a normal Parla function and can be called as such.


### Variables

```python
x
```
(expression)
A variable reference.

```python
x = e
```
(statement)
Store the value of expression in the variable `x`.
Variables in Parla are implicitly references to values instead of storing them.
So, `x = e_2` will cause `x` to reference the new values, but will not modify the value originally referenced by `x` in any way.

Each variable must have an unique initial assignment which dominates all its uses and all following assignments must be of the same type (or a subtype).
The initial assignment acts as a declaration and provides an implicit type.
Therefor, a variable is not allowed to be assigned a conditional branch (if-statement or while-loop) without being assigned an initial value before the branch.

**TODO: How are Ref arguments handled? Implicitly creating a ref? Explicitly?**


## Array Storage Annotations

**TODO: This section should move to a Type implementations file or section in Types.md**

The interface for arrays does not specify the storage.
However the storage can be controlled by these annotations.
They are written as methods on arrays that make copies, but as an optimization expressions which apply these functions to an array constructor directly, such as `Array(T)(10, 40).majority(1, 0).structure_of_arrays()`, are compiled to code that directly generates the final array.
Similarly, a sequence of storage annotations is collapsed together into a single copy operation.


### Dimension Ordering

```python
e_1.majority(i_1, ..., i_n)
```
(expression; `e_1: Array(T), i_j: Dimension -> Array(T)`)
Create a copy of `e_1` such that the *storage* of the dimensions (identified by number) is ordered as specified by the arguments.
This does **not** change the order of the dimensions in indexing expressions; it only changes how the underlying data is stored.
This allows switching between row-major and column-major order.
*Example:* If `a[row, col]` is an array then `a.majority(1, 0)` is a copy which is stored (but not indexed) in `col`-major order regardless of its previous storage.

**FIXME: The name `majority` is bad. "Dimension order"?**


### Structure of Arrays vs. Array of Structures

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

**TODO: Try to infer types in simple cases. Return type should be fairly easy. But not that useful.**

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


### Parallelism Restriction Annotations


```python
with lock(e):
    statements
```
(statement; `e : Lock`)
Execute `statements` with the `Lock` `e` held on the current device.
Lock state is not shared across devices so locks only provide mutual exclusion within devices.


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
An expression in Parla.
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


