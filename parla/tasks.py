"""
Parla supports simple task parallelism.
"""

from contextlib import contextmanager

@contextmanager
def spawn():
    """
    Execute the body of the `with` block as a new task.
    The task will execute in parallel with any following code.

    >>> with spawn():
    >>>     ...

    """
    yield

@contextmanager
def finish():
    """
    Execute the body of the `with` normally and then perform a barrier applying to all tasks created (statically scoped).
    This block has the same semantics as the implicit barrier on a function in Cilk.

    >>> with finish():
    >>>     ...

    """
    yield
