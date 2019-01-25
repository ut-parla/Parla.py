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

Implementation
--------------

We will finally implement Parla as an embedded language in Python.
The implementation will be based on the :std:doc:`Numba <numba:user/overview>` compiler for array-based Python programs.
We will extend Numba to support the Parla API.
The runtime for Parla will use `Dynd <http://libdynd.org/>`_ to provide low-level array types and operations.
The runtime may also use `Galois's runtime library <http://iss.ices.utexas.edu/?p=projects/galois>`_ to handle parallel tasks and threads.
Other parallelism runtimes could be supported (for instance, the OpenMP runtime, to allow Parla code to interoperate with OpenMP code).

This Parla implementation will initially provide Just-in-Time (JIT) compilation only.
Later versions could support Ahead-of-Time (AOT) compilation and even calling into Parla from C, both using :std:doc:`Numba's AOT support <numba:user/pycc>`.
