"""
Parla flow-control is similar to Python except that loops are parallel by default (except for `while`).
To restrict this parallelism, Parla provides `reads` and `writes` annotations from which dependencies can be generated.
Sequential loops are created by using one of the iterable constructors in `seq`.
Parla while loops are identical to Python's.

Parla `parallel iterators <ParIterator>` accept hints on how to perform the iteration.

This entire module should be imported to allow it to override the builtin `~builtins.range` and `~builtins.iter` operations.

>>> from parla.loops import *

.. note:: All the looping constructs are single dimensional. For multi-dimensional operations nest loops.
.. note:: As in Python, `with` blocks can be combined as follows: `with reads(v), writes(w): ...`

.. todo:: Should parallel loops have an implicit barrier for iterations at loop exit? :func:`parla.tasks.finish`

.. testsetup::

    code = None
    e = [0]
    a = [0]
    i = 0
"""

from __future__ import annotations

import builtins
from contextlib import contextmanager

class seq:
    """
    Sequential iteration for Parla.

    Sequential range iteration:

    >>> for x in seq.range(0, 10, 1):
    ...     code

    Sequential iteration over an interable:
    
    >>> for x in seq.iter(e):
    ...     code
    """
    range = builtins.range
    iter = builtins.iter


class ParIterator:
    def __init__(self, underlying):
        self._underlying = underlying
    
    def __iter__(self):
        return self
    def __next__(self):
        return next(self._underlying)

    def hint(self, *, vectorize : int = None) -> ParIterator:
        """
        Provide compilation hints and requests to the compiler.
        The compiler will produce (optional) warnings if the hints are not followed.
        However, even if the Parla compiler follows the hints, the target backend may change the resulting code.

        This method invalidates `self`.
        
        :param vectorize: Request that the compiler vectorize the loop assuming the `vectorize` lanes.
        :return: A hinted iterator based on `self`.
        """
        return self
        

def range(start, stop = None, step = 1, **hints) -> ParIterator:
    """
    A parallel iterable range.
    All iterations execute in parallel.

    >>> for x in range(0, 10, 2):
    ...     code

    :param \*\*hints: Any hints accepted by `ParIterator.hint` can be passed to `range` as keyword arguments.
    """
    return ParIterator(builtins.iter(builtins.range(start, stop, step))).hint(**hints)


def iter(iterable, **hints) -> ParIterator:
    """
    A parallel for-each loop over each element of the array `iterable`.

    >>> for x in iter(a):
    ...     code

    This is similar to: 

    >>> for i in range(e.size(0)): 
    ...     x = e[i, ...]
    ...     code

    .. todo:: What dimension should iter use on arrays? Inner most or outer most?

    :param \*\*hints: Any hints accepted by `ParIterator.hint` can be passed to `range` as keyword arguments.
    """
    return ParIterator(builtins.iter(iterable)).hint(**hints)


def reduce(iterable : ParIterator or iterable, operator):
    """
    Reduce in parallel over the given `iterable`.
    The iterable will generally be a parallel generator expression (see below).
    The generator must be `~parla.function_decorators.pure`.
    The reduction :py:data:`operator` is called with two values and must be `~parla.function_decorators.associative` and `~parla.function_decorators.pure`.

    >>> reduce((a[i, i] for i in range(0, 100)), op)

    Associativity of the reduction operation allows any arbitrary reduction tree to be used for the computation.
    If :py:data:`operator` is annotated as `~parla.function_decorators.commutative`, `reduce` can perform reductions on any available values without concern for which index or indicies they represent.
    This commutative reduce is more efficient on some devices.

    Since all the functions used in the reduction are side-effect free, `reads` and `writes` will never affect reductions.

    :param iterable: An iterable to reduce.
    :type iterable: `ParIterator` or, in rare cases, any iterable
    :param operator: The associative reduction operator.
    :type operator: A function of type `(T, T) => T`
    
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
    ...     code

    """
    yield

@contextmanager
def writes(v):
    """
    Tell the compiler that the bosy writes the value `v` and these writes must occur as if all statically enclosing loops were sequential.
    The same restrictions as for `reads` apply to this.

    >>> with writes(a[i]):
    ...     code

    """
    yield

