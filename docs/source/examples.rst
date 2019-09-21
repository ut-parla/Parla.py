Examples
========


Inner Product
----------------

This example shows a simple implementation of inner product.

.. literalinclude:: /../../examples/inner.py
    :pyobject: main

.. _Fox's Algorithm:

Mat–Vec by Fox's Algorithm
---------------------------

This example shows matrix–vector multipy implemented using Fox's algorithm.
The `main` function partitions the data, executes, two multiplies, and then collects the result back to the CPU.

.. literalinclude:: /../../examples/fox.py
    :pyobject: partition_fox
.. literalinclude:: /../../examples/fox.py
    :pyobject: matvec_fox_partitioned
.. literalinclude:: /../../examples/fox.py
    :pyobject: collect_fox
.. _ main:
.. literalinclude:: /../../examples/fox.py
    :pyobject: main