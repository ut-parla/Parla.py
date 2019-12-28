#!/bin/bash

ncpu=1
ngpu=4

for i in $(seq 0 $ngpu);
do
	rm -rf "kokkos/gpu${i}"
	cp -R dev "kokkos/gpu${i}"
	python gpu_n_setup.py build_ext --inplace -n $i
done

for i in $(seq 0 $ncpu);
do
	rm -rf "kokkos/cpu${i}"
	cp -R dev "kokkos/cpu${i}"
	python cpu_n_setup.py build_ext --inplace -n $i
done
