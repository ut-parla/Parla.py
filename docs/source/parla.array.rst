parla.array
===========


.. testsetup::

   from __future__ import annotations
   from parla.array import *
   
.. automodule:: parla.array
   :no-members:


Constructors
------------

.. autofunction:: zeros
.. autofunction:: filled
.. autofunction:: ref


Array[T, k]
-----------

.. autoclass:: Array

.. autodata:: newdim
.. autoclass:: StorageLayout
               
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

ImmutableInplaceArray[T, s₁, …, sₖ]
***********************************

.. autoclass:: ImmutableInplaceArray

MutableInplaceArray[T, s₁, …, sₖ]
*********************************

.. autoclass:: MutableInplaceArray
