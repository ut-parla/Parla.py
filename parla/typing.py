from __future__ import annotations

import inspect
import types as pytypes
from typing import Tuple, List

__all__ = ["TypeVar", "LiftedNaturalVar", "TypeConstructor", "GenericFunction", "GenericClassAlias", "type_to_str", "Tuple", "List"]

def type_to_str(arg):
    if isinstance(arg, type):
        return arg.__name__
    else:
        return str(arg)


class TypeVar(type):
    def __new__(cls, name):
        return super().__new__(cls, name, (), {})

    def __init__(self, name):
        pass

class LiftedNaturalVar(TypeVar):
    pass

class TypeConstructor:
    @classmethod
    def _new_application(cls, constr, args):
        name = "{:s}[{}]".format(constr.__name__, ", ".join(type_to_str(t) for t in args))
        def body(ns):
            ns["type_constructor"] = constr
            ns["args"] = args
        return pytypes.new_class(name, (constr,), exec_body=body)
    
    def __class_getitem__(cls, params):
        return cls._new_application(cls, params)


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
