import inspect as pythoninspect
import functools
import collections

__all__ = [
    "VariantDefinitionError", "device_specialized",
    "has_property",
    "side_effect_free",
    "pure",
    "associative",
    "commutative"
]

class VariantDefinitionError(ValueError):
    """
    A function variant definition is invalid.

    :see: `device_specialized`
    """
    pass

class _DeviceSpecializer:
    def __init__(self, func):
        self._default = func
        self._variants = {}
        functools.update_wrapper(self, func)

    def variant(self, *ts):
        if any(t in self._variants for t in ts):
            raise VariantDefinitionError("variant({}) is already defined for {name}".format(target, name = self._default.__name__))
        def variant(f):
            for t in ts:
                self._variants[t] = f
        variant.__name__ = "{}.variant".format(self._default.__name__)
        return variant

    def __call__(self, *args, **kwds):
        return self._default(*args, **kwds)

    def get_variant(self, target):
        return self._variants.get(target, self._default)

    def __repr__(self):
        return "{f} specialized to {targets}>".format(f = repr(self._default)[:-1], targets = tuple(self._variants.keys()))


def device_specialized(f):
    """
    A decorator to declare that this function has specialized variants for specific devices.
    The decorated function is the default implemention.

    To provide a specialized variant use the `variant` member of the main function:

    .. testsetup::

        from parla.function_decorators import *

    >>> @device_specialized
    ... def f():
    ...     ...
    >>> @f.variant('DEVICE')
    ... def f_gpu():
    ...     ...

    `DEVICE` above will often by something like `HOST` or `GPU`, but is extensible.
    Multiple devices can be specified as separate parameters to use the same implementation on multiple devices: `@f.variant(CPU, FPGA)`.
    Each device can only be used once on a given function.

    Device specialized functions are called just like any other function, but the implementation which is called is selected based on where the code executes.
    The compiler will make the choice when it is compiling for a specific target.

    .. TODO::
        This is **not** implement in the :ref:`Parla Prototype`.
    """
    return _DeviceSpecializer(f)

def _get_properties(f) -> set:
    try:
        return f.__function_propreties__
    except AttributeError:
        f.__function_propreties__ = set()
        return f.__function_propreties__

def _is_proprerty(prop):
    return pythoninspect.isfunction(prop) and prop.__module__ == "parla.function_decorators"

def side_effect_free(f):
    """
    A decorator to declare that a function has no side effects.
    """
    _get_properties(f).add(side_effect_free)
    return f

def pure(f):
    """
    A decorator to declare a function as *pure*.
    Where pure means:

    * Its return value depends only on its parameters and not on any internal or external state; and
    * Its evaluation has no side effects.
    """
    _get_properties(f).add(pure)
    return side_effect_free(f)

def associative(f):
    """
    A decorator to declare that a function is associative, meaning applying the function to reduce a subset of the arguments ahead of this call will not change the result.
    This is a simple generalization of associativity of binary operators.

    """
    _get_properties(f).add(associative)
    return f

def commutative(f):
    """
    A decorator to declare that a function is commutative, meaning the order of arguments does not matter.
    """
    _get_properties(f).add(commutative)
    return f

def has_property(f, prop):
    """
    Check if a callable has a specific :ref:`property <Properties>`.
    Generally the property is only known if the callable is decorated with `prop`.

    :param f: A function or method.
    :param prop: A property decorator, such as `pure`.

    :return: True iff `f` has `prop`
    """
    if not _is_proprerty(prop):
        raise TypeError("{} is not a function property".format(prop))
    return prop in _get_properties(f)
