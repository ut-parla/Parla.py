# Languages

This directory contains documentation for the languages for our PSAAP proposal: Parla (surface language) and Gergo (core language).
The goal of Parla is to provide a simple but usable set of high-performance programming features and allow automatic optimizations while still allowing programming control in all cases.
Parla should be usable by real world programmers.
The Parla documentation/specification is written as a well documented Python library.
The Python files are in the package parla in this directory and additional documentation files are in docs.
The documentation is built with Sphinx.
A prebuilt version is maintained at: http://www.cs.utexas.edu/~amp/psaap/Parla.py/index.html

The goal of Gergo is to provide a *minimal* but expressive set of primitives to describe and optimize Parla programs.
Gergo will not be usable by programmers, but will be simpler and more uniform to make optimization easier.
Gergo also doesn't have any documentation and is almost entirely undefined.
