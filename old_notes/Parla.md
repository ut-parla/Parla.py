# Parla

Parla uses a python like syntax which we will tweak to make it actually valid Python code for the prototype (and maybe the final version).


## Core Primitives


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


