Tasks (parla.tasks)
===================

.. testsetup::

   from __future__ import annotations
   from parla.tasks import *

.. automodule:: parla.tasks
   :no-members:


Annotations
-----------

.. autofunction:: spawn

Tasks and Groups of Tasks
-------------------------

All tasks, task IDs, and task sets are :term:`awaitable`. Awaiting one will block until the tasks have completed.

.. autoclass:: Task
   :no-members:
   :members: __await__
.. autoclass:: TaskID
   :no-members:
   :members: task, id, __await__
.. autoclass:: TaskSpace
   :no-members:
   :members: __getitem__, __await__
.. autoclass:: tasks
   :no-members:
   :members: __await__
.. autoclass:: CompletedTaskSpace
   :no-members:
   :members: __await__

Utilities
---------

.. autofunction:: finish
.. autofunction:: get_current_devices

