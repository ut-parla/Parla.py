"""
Parla allows programmers to require or hint that the compiler or runtime should perform certain optimizations or use certain storage layouts.
These are specified as *details*.

The compiler will produce errors if requirements are not followed and (optional) warnings if the hints are not followed.
However, even if the Parla compiler follows the hints, the target backend may change the resulting code.

This module provides a simple framework for declaring, applying, and accessing details.
The Parla API consistently uses `DetailableType` and `Detailable` to accept hints and requirements.
So any requirments specifications should follow one of these patterns:

.. code-block:: python

  Type.require(detail(arguments, ...), ...)
  value.require(detail(arguments, ...), ...)

or similarly for `hint`.
hint and require can be chained.
"""

import types as pytypes

class Detail:
    """
    The superclass of all details.
    Details should be declared as a subclass.
    If the detail takes parameters, the subclass must override `__init__`.
    """
    @classmethod
    def get_hint(cls, obj):
        """
        Get the instance of this detail attached to `obj` as a hint.

        :param obj: An object with details.
        :return: The detail instance or None.
        """
        return next((d for d in getattr(obj, "__hints__", ()) if isinstance(d, cls) or d == cls), None)

    @classmethod
    def get_requirement(cls, obj):
        """
        Get the instance of this detail attached to `obj` as a requirement.

        :param obj: An object with details.
        :return: The detail instance or None.
        """
        return next((d for d in getattr(obj, "__requirements__", ()) if isinstance(d, cls) or d == cls), None)

    def __str__(self):
        return "{}({})".format(type(self).__name__, ", ".join(getattr(self, "args", ())))

class IntDetail(Detail):
    def __init__(self, v: int):
        self.args = (v,)
        self.value = v

class DetailUnsupportedError(TypeError):
    """
    An exception raise if a detail is applied to a type or value that doesn't support it.
    """
    pass

class Detailable:
    """
    A mix-in class that adds support for hinting to instances.

    The `__instance_details__` of each subclass should include all detail types which instances can accept.
    """

    # TODO: Replace old details of the same type.

    __instance_details__ = frozenset()

    def _check_details(self, allowed, actual):
        if not all(d in allowed for d in actual):
            raise DetailUnsupportedError("Detail not supported. Supported details are: {}".format(", ".join(str(c) for c in allowed)))

    def hint(self, *args):
        """
        Apply hints to `self` and return a value that should be used in place of it.
        The new value might be `self`, but it might also be a view of some kind or some other replacement object.

        :allocation: Never
        """
        self._check_details(self.__instance_details__, args)
        self.__hints__ += args
        return self

    def require(self, *args):
        """
        Apply requirements to `self` and return a value that should be used in place of it.
        The new value might be `self`, but it might also be a view of some kind or some other replacement object.

        :allocation: Never
        """
        self._check_details(self.__instance_details__, args)
        self.__requirements__ += args
        return self

class DetailableType(Detailable):
    """
    A mix-in class that adds support for hinting to the type (class) object itself and on instances.

    The `__type_details__` of each subclass should include all detail types which the subclass can accept.

    .. note::
      This type has separate class methods `hint` and `require` and inherited instance methods `Detailable.hint` and `~Detailable.require`.
      This is enabled by a bit of a hack.
    """

    __type_details__ = frozenset()

    def __init__(self, *args, **kwds):
        self.hint = Detailable.hint
        self.require = Detailable.require
        self.__hints__ = ()
        self.__requirements__ = ()
        super().__init__(*args, **kwds)

    @classmethod
    def hint(cls, *args):
        """
        :return: a new subclass of this class with the hints added.
        """
        self._check_details(cls.__type_details__, args)
        def body(ns):
            ns["__hints__"] = getattr(cls, "__hints__", ()) + args
        return pytypes.new_class(cls.__name__, (cls,), exec_body=body)

    @classmethod
    def require(self, *args):
        """
        :return: a new subclass of this class with the requirements added.
        """
        self._check_details(cls.__type_details__, args)
        def body(ns):
            ns["__requirements__"] = getattr(cls, "__requirements__", ()) + args
        return pytypes.new_class(cls.__name__, (cls,), exec_body=body)
