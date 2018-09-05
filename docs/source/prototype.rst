.. _`Parla Prototype`:

Parla Prototype
===============

The Parla prototype implementation will go through several phases before being a complete implementation.
Each section on this page discusses one of those phases.

.. _`syntax markup`:

Syntax Mockup
-------------

As an initial way to explore the use of Parla, we will mockup the Parla API using Python stub functions.
This will allow Parla programs to parse in Python (to check the syntax at least) and will provide detailed API and language documentation (see :mod:`parla`).

Python Implementation
---------------------

We will provide a rough implementation for the :ref:`syntax markup` API.
This implementation will be able execute Parla programs (at least simple ones) and verify that we are not missing information needed for an implementation.
Initially, this implementation will be totally sequential (expect potentially inside `~parla.array` operations or foreign calls).
However, a parallel implementation would be possible with additional work.

.. note::
   A parallel implementation would require Python AST rewriting which would probably be fragile (break in the presences of particular language features or variable usage patterns), but it would not be all that difficult.
   An `@parla` annotation would find all parallel `for` loops and rewrite them to a body function definition and a call to `parla.loops.forloop(iterable, body)` which would implement parallel for loops.
          
Compiled Implementation
-----------------------

We will finally implement Parla as an embedded language in Python.
The implementation will be based on the :std:doc:`Numba <numba:user/overview>` compiler for array-based Python programs.
We will extend Numba to support the Parla API.
The runtime for Parla will use `Dynd <http://libdynd.org/>`_ to provide low-level array types and operations.
The runtime may also use `Galois's runtime library <http://iss.ices.utexas.edu/?p=projects/galois>`_ to handle parallel tasks and threads.
Other parallelism runtimes could be supported (for instance, the OpenMP runtime, to allow Parla code to interoperate with OpenMP code).

This Parla implementation will initially provide Just-in-Time (JIT) compilation only.
Later versions could support Ahead-of-Time (AOT) compilation and even calling into Parla from C, both using :std:doc:`Numba's AOT support <numba:user/pycc>`.

