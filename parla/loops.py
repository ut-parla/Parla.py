"""
Parla flow-control is similar to Python except that loops are parallel by default (except for `while`).
Parla while loops are identical to Python's.

Parla's parallel loops assume that all iterations are independent and do not have an implicit barrier at the end of the loop.
If this is not the case the programmer must explicitly restrict the parallelism.
To restrict this parallelism, Parla provides `reads` and `writes` annotations from which dependencies can be generated.
Since Parla loops don't have an implicit barrier, the programmer should use :func:`with finish():<parla.tasks.finish>` around the loop to guarantee that loop iterations are finished if that is required.

Sequential loops are created by using one of the iterable constructors in `seq`.

Parla `parallel iterators <ParIterator>` accept hints on how to perform the iteration.

This entire module should be imported to allow it to override the builtin `~builtins.range` and `~builtins.iter` operations.

>>> from parla.loops import *

.. note:: All the looping constructs are single dimensional. For multi-dimensional operations nest loops.
.. note::
  As in Python, `with` blocks can be combined as follows: 

  .. code-block:: python

      with reads(v), writes(w): ...


.. testsetup::

    code = None
    e = [0]
    a = [0]
    i = 0
"""

from __future__ import annotations

import builtins
from contextlib import contextmanager

from parla.tasks import finish
from parla import detail

__all__ = [
    "seq",
    "range", "iter",
    "reduce",
    "reads", "writes",
    "finish"
]

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


class vectorize(detail.IntDetail):
    """
    Request that the compiler vectorize the loop for the specified number of lanes lanes.

    >>> for i in range(100).vectorize(4):
    >>>     code
    
    """
    pass
    
    
class ParIterator(detail.Detailable):
    """
    An iterator which executes loops in parallel.
    """
    def __init__(self, underlying):
        self._underlying = underlying
    
    def __iter__(self):
        return self
    def __next__(self):
        return next(self._underlying)

    def hint(self, *args):
        """
        Return a new ParIterator with hints.
        
        This invalidates `self`.

        :return: a new iterator
        """
        return super().hint(*args)

    def require(self, *args):
        """
        Return a new ParIterator with hints.
        
        This invalidates `self`.

        :return: a new iterator
        """
        return super().require(*args)
        

def range(start, stop = None, step = 1, *, requirements=(), hints=()) -> ParIterator:
    """
    A parallel iterable range.
    All iterations execute in parallel.

    >>> for x in range(0, 10, 2):
    ...     code

    :param requirements: Any requirements accepted by `ParIterator`.
    :param hints: Any hints accepted by `ParIterator`.
    """
    return ParIterator(builtins.iter(builtins.range(start, stop, step))).hint(**hints)


def iter(iterable, *, requirements=(), hints=()) -> ParIterator:
    """
    A parallel for-each loop over each element of the array `iterable`.

    >>> for x in iter(a):
    ...     code

    Executes similarly to: 

    >>> for i in range(e.shape[0]): 
    ...     x = e[i, ...]
    ...     code

    :param requirements: Any requirements accepted by `ParIterator`.
    :param hints: Any hints accepted by `ParIterator`.

    .. seealso:: :meth:`Array.__iter__ <parla.array.Array.__iter__>`
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

