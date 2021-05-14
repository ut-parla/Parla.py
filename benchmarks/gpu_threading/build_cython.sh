#!/bin/bash

rm -rf test
cp -R dev test
python gpu_setup.py build_ext --inplace --force -n 0

#rm -rf "kokkos/cpu"
#cp -R dev "kokkos/cpu"
#python cpu_setup.py build_ext --inplace --force -n 0
