import functools

from . import tasks

__all__ = [
    "VariantDefinitionError", "specialized",
]


class VariantDefinitionError(ValueError):
    """
    A function variant definition is invalid.

    :see: `specialized`
    """
    pass


class _ArchitectureSpecializer:
    def __init__(self, func):
        self._default = func
        self._variants = {}
        functools.update_wrapper(self, func)

    def variant(self, *ts):
        if any(t in self._variants for t in ts):
            raise VariantDefinitionError(
                "variant({}) is already defined for {name}".format(target, name=self._default.__name__))

        def variant(f):
            for t in ts:
                self._variants[t] = f
            return self

        variant.__name__ = "{}.variant".format(self._default.__name__)
        return variant

    def __call__(self, *args, **kwds):
        # TODO: Get current device and use it to select the varient
        d = tasks.get_current_device()
        f = self.get_variant(d.architecture)
        return f(*args, **kwds)
        # return self._default(*args, **kwds)

    def get_variant(self, target):
        return self._variants.get(target, self._default)

    def __repr__(self):
        return "{f} specialized to {targets}>".format(f=repr(self._default)[:-1], targets=tuple(self._variants.keys()))


def specialized(f):
    """
    A decorator to declare that this function has specialized variants for specific architectures.
    The decorated function is the default implemention, used when no specialized implementation is available.
    The default can just be `raise NotImplementedError()` in cases where no default implementation is possible.

    To provide a specialized variant use the `variant` member of the main function:

    .. testsetup::

        from parla.function_decorators import *

    >>> @specialized
    ... def f():
    ...     raise NotImplementedError()
    >>> @f.variant(architecture)
    ... def f_gpu():
    ...     ...

    `architecture` above will often by something like `cpu` or `gpu`, but is extensible.
    Multiple architectures can be specified as separate parameters to use the same implementation on multiple architectures: `@f.variant(CPU, FPGA)`.
    Each architecture can only be used once on a given function.

    Architecture specialized functions are called just like any other function, but the implementation which is called is selected based on where the code executes.
    The compiler will make the choice when it is compiling for a specific target.
    """
    return _ArchitectureSpecializer(f)
