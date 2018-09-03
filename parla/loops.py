"""
Parla flow-control is similar to Python except that loops are parallel by default (except for `while`).
To restrict this parallelism, Parla provides `reads` and `writes` annotations from which dependencies can be generated.
Sequential loops are created by using one of the iterable constructors in `seq`.
Parla while loops are identical to Python's.

>>> while e:
>>>     ...

This entire module should be imported to allow it to override the builtin `~builtins.range` and `~builtins.iter` operations.

>>> from parla.loops import *

.. note:: All the looping constructs are single dimensional. For multi-dimensional operations nest loops.
.. note:: As in Python, `with` blocks can be combined as follows: `with reads(v), writes(w): ...`

.. todo:: parallel iteration annotations: vectorization
.. todo:: Should parallel loops have an implicit barrier for iterations at loop exit? :func:`parla.tasks.finish`
"""

import builtins
from contextlib import contextmanager

class seq:
    """
    Sequential iteration for Parla.

    Sequential range iteration:

    >>> for x in seq.range(0, 10, 1):
    ...     ...

    Sequential iteration over an interable:
    
    >>> for x in seq.iter(e):
    ...     ...
    """
    range = builtins.range
    iter = builtins.iter

def range(start, stop = None, step = 1):
    """
    A parallel iterable range.
    All iterations execute in parallel.

    >>> for x in range(0, 10, 2):
    ...     ...
    """
    return builtins.range(start, stop, step)


def iter(iterable):
    """
    A parallel for-each loop over each element of the array `iterable`.

    >>> for x in iter(a):
    ...     ...

    This is similar to: 

    >>> for i in range(e.size(0)): 
    ...     x = e[i, ...]
    ...     ...
    """
    return builtins.iter(iterable)


def reduce(iterable, operator):
    """
    Reduce in parallel over the given `iterable`.
    The iterable will generally be a parallel generator expression (see below).
    The generator must be `~parla.function_decorators.pure`.
    The reduction :py:data:`operator` is called with two values and must be `~parla.function_decorators.associative` and `~parla.function_decorators.pure`.

    >>> reduce((a[i, i] for i in range(0, 100, 1)), op)

    Associativity of the reduction operation allows any arbitrary reduction tree to be used for the computation.
    If :py:data:`operator` is annotated as `~parla.function_decorators.commutative`, `reduce` can perform reductions on any available values without concern for which index or indicies they represent.
    This commutative reduce is more efficient on some devices.

    Since all the functions used in the reduction are side-effect free, `reads` and `writes` will never affect reductions.
    """
    result = None
    for v in iterable:
        if result is None:
            result = v
        else:
            result = operator(result, v)
    return result
            


@contextmanager
def reads(v):
    """
    Tell the compiler that the body must read the value of `v` as if all statically enclosing loops were executed sequentially.
    `v` must be a variable reference or a simple indexing expression into an array or static array view.
    A static array view is one which is defined in the current function using only constants.

    >>> with reads(a[i]):
    >>>     ...

    """
    yield

@contextmanager
def writes(v):
    """
    Tell the compiler that the bosy writes the value `v` and these writes must occur as if all statically enclosing loops were sequential.
    The same restrictions as for `reads` apply to this.

    >>> with writes(a[i]):
    >>>     ...

    """
    yield

