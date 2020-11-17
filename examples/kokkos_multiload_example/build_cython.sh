#!/bin/bash

rm -rf "kokkos/gpu"
cp -R dev "kokkos/gpu"
python gpu_setup.py build_ext --inplace -n 0

rm -rf "kokkos/cpu"
cp -R dev "kokkos/cpu"
python cpu_setup.py build_ext --inplace -n 0
