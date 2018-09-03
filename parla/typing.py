from __future__ import annotations

import inspect

__all__ = ["TypeVar", "LiftedNaturalVar", "TypeConstructor", "GenericFunction", "GenericClassAlias", "type_to_str"]

def type_to_str(arg):
    if isinstance(arg, type):
        return arg.__name__
    else:
        return str(arg)


class TypeVar:
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return self.name

class LiftedNaturalVar:
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return self.name

class _TypeApplication:
    def __init__(self, cls, args):
        self.cls = cls
        self.args = args

    def __mro_entries__(self, bases):
        return (self.cls,)

    def __repr__(self):
        return "{:s}[{}]".format(self.cls.__name__, ", ".join(type_to_str(t) for t in self.args))

class TypeConstructor:
    def __class_getitem__(cls, params):
        return _TypeApplication(cls, params)


## Patch isfunction to make pydoc behave correctly

_old_isfunction = inspect.isfunction

class GenericFunction:
    pass

def _isfunction(f):
    return _old_isfunction(f) or isinstance(f, GenericFunction)

inspect.isfunction = _isfunction

## Patch isclass to make pydoc behave correctly

_old_isclass = inspect.isclass

class GenericClassAlias:
    pass

def _isclass(f):
    return _old_isclass(f) or isinstance(f, GenericClassAlias)

inspect.isclass = _isclass
