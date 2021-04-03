"""
Generic, standalone utilities.

All the stuff here should NOT depend on any other Parla-specific types/functions/etc.
"""
from typing import TypeVar, Tuple, Union, Iterable, Callable

__all__ = ["traverse",]


# TODO (bozhi): another parla module for typing?
T = TypeVar('T') # (PEP 484) The argument to TypeVar() must be a string equal to the variable name to which it is assigned.


def traverse(prefix: T, index: Tuple[Union[slice, Iterable[int], int]], step: Callable[[T, int], T], stop: Callable[[T], None]):
    """Traverse the :param:`index`, update :param:`prefix` by applying :param:`step`, :param:`stop` at leaf calls.
    
    :param prefix: the initial state
    :param index: the index tuple containing subindexes
    :param step: a function with 2 input arguments (current_state, subindex) which returns the next state, applied for each subindex.
    :param stop: a function with 1 input argument (final_state), applied each time subindexes exhaust.

    .. todo::
        Used to be nested in TaskSpace.__getitem__ and now needs a better function name.
    """
    if len(index) > 0:
        i, *rest = index
        if isinstance(i, slice):
            for v in range(i.start or 0, i.stop, i.step or 1):
                traverse(step(prefix, v), rest, step, stop)
        elif isinstance(i, Iterable):
            for v in i:
                traverse(step(prefix, v), rest, step, stop)
        else:
            traverse(step(prefix, i), rest, step, stop)
    else:
        stop(prefix)
