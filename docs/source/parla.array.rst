parla.array
===========


.. testsetup::

   from __future__ import annotations
   from parla.array import *
   
.. automodule:: parla.array
   :no-members:


Functions
---------

.. autofunction:: zeros
.. autofunction:: filled
.. autofunction:: freeze
.. autofunction:: ref
.. autofunction:: deref


Array[T, k]
-----------

.. autoclass:: Array

ImmutableArray[T, k]
********************

.. autoclass:: ImmutableArray


MutableArray[T, k]
******************

.. autoclass:: MutableArray

Ref[T]
------

.. autoclass:: Ref

ImmutableRef[T]
***************

.. autoclass:: ImmutableRef

MutableRef[T]
*************

.. autoclass:: MutableRef

InplaceArray[T, s₁, …, sₖ]
--------------------------

.. autoclass:: InplaceArray

InplaceImmutableArray[T, s₁, …, sₖ]
***********************************

.. autoclass:: InplaceImmutableArray

InplaceMutableArray[T, s₁, …, sₖ]
*********************************

.. autoclass:: InplaceMutableArray

Escape-Hatch Functions
----------------------

.. autofunction:: UNSAFE_freeze
