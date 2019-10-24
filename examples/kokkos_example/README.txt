How to Build
---------------------

1) Download Kokkos from https://github.com/kokkos/kokkos (use the develop branch)
2) Set KOKKOS_DIR as a path to the root kokkos folder (e.g. export KOKKOS_DIR=~/kokkos)
3) Run build_kokkos.sh

4) Make sure you have a python enviornment that supports cython and parla
5) Run build_cython.sh while in the kokkos_example directory

Our simple kokkos cython extension is now built and can be used. 

6) Run the example: run_parla_example.py

How to Modify
-------
If you want to change the kokkos function only three files in dev must be updated:
dev/kokkos_compute.hpp - Containing the kokkos implementation
dev/core.pxd           - Containing a header for the kokkos implementation
dev/core.pyx           - Containing the details of the python extension 

Additionally, for convienence in dispatching to the correct device we have:
kokkos/core.py - which provides wrappers to dispatch to specializations

Under Construction
----------------------
- Improvement: Get deviceIDs from Parla instead of reading them from CuPy

