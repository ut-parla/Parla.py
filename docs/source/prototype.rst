.. _`Parla Prototype`:

Parla Prototype
===============

The Parla prototype implementation will go through several phases before being a complete implementation.
Each section on this page discusses one of those phases.

Python Prototype
----------------

This implementation will be able execute Parla programs (at least simple ones) and verify that we are not missing information needed for an implementation.
Initially, this implementation will not be truly parallel in orchestration code (external operations, e.g. numpy operations, will execute truly in parallel).
The runtime is based on Galois.

This prototype has two primary restrictions:

1. The orchestration must be entirely inside a single top-level task and that task will be executed *synchronously*.
2. Orchestration code is run in the Python interpreter and therefore for not perform all that well. This restriction does not affect external operations like numpy operations or BLAS calls.

These limitations will be removed as we move towards a complete implementation.

Physical Placement of Data and Computation
------------------------------------------

For initial experimentation, Parla provides ways to place data and computation on specific physical devices.

To copy data into a location, apply a memory detail to that value:

.. testsetup::
    import numpy as np

.. code-block:: python

    a = np.array([1, 2, 3])
    # Copy the array to GPU #0.
    b = gpu(0).memory()(a)
    # Or identically
    g0mem = gpu(0).memory()
    b = g0mem(a)

The new object `b` is not related to the original array `a`.
The user will have to perform manual copies to keep them in sync if needed.

To place tasks at a location, provide the `placement` argument to `~parla.tasks.spawn`:

.. code-block:: python

    @spawn(T2[j], [T1[j, 0:j]], placement=gpu(j%4))
    def t2():
        cholesky_inplace(a[j,j])

The task will be run on the specific device provided and no other.
As always, the task orchestration code will run on the CPU, but appropriate `~parla.function_decorators.specialized` function variant will be used and those should be setup to call device kernels.
