"""
Parla supports simple task parallelism.

.. testsetup::

    code = None

"""

from contextlib import contextmanager

@contextmanager
def spawn():
    """
    Execute the body of the `with` block as a new task.
    The task will execute in parallel with any following code.

    >>> with spawn():
    ...     code

    """
    yield

@contextmanager
def finish():
    """
    Execute the body of the `with` normally and then perform a barrier applying to all tasks created.
    This block has the similar semantics to the ``sync`` in Cilk.

    >>> with finish():
    ...     code

    """
    yield
