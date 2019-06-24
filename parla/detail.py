"""
Parla allows programmers to instruct the compiler or runtime to perform certain optimizations or use certain storage layouts.
These are specified as *details*.
This module provides a simple framework for declaring, applying, and accessing details.

To apply a detail to a target value call the detail with the target as an argument and use the returned value in place of the target.
So, any requirements specifications should follow one of these patterns:

.. code-block:: python

  target = detail(arguments...)(target)
  direct_usage(detail(arguments...)(target))


"""

import warnings


class Detail:
    """
    The superclass of all details.
    Details should be declared as a subclass.
    If the detail takes parameters, the subclass must override `__init__`.
    """

    @classmethod
    def get(cls, obj):
        """
        Get the instance of this detail attached to `obj`.

        :param obj: An object with details.
        :return: The detail instance or None.
        """
        return next((d for d in getattr(obj, "__details__", ()) if isinstance(d, cls) or d == cls), None)

    def __call__(self, target):
        """
        Apply this detail to `target`.

        The default implementation adds `self` to the attribute `__details__` and returns `target`.

        Subclasses may override this method to attach the appropriate information to `target` or wrap `target`.
        """
        try:
            details = getattr(target, "__details__", [])
            details.append(self)
            target.__details__ = details
        except (TypeError, AttributeError) as e:
            warnings.warn("Detail applied to unsupported type. Override __call__ in {}. Or this might be a user code "
                          "bug.".format(type(self)), DeprecationWarning)
        return target

    def __str__(self):
        return "{}({})".format(type(self).__name__, ", ".join(getattr(self, "args", ())))


class DetailUnsupportedError(TypeError):
    """
    An exception raise if a detail is applied to a type or value that doesn't support it.
    """
    pass
