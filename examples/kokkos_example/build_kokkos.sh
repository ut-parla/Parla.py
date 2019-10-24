#!/bin/bash


cd "${KOKKOS_DIR}"

rm -rf gpu_build
mkdir gpu_build
cd gpu_build

../generate_makefile.bash --cxxflags="-fPIC" --ldflags="-fPIC" --arch=Pascal61 --prefix=../lib --with-cuda --with-cuda-options="enable_lambda"

make kokkoslib
make install

cd ..

rm -rf cpu_build
mkdir cpu_build
cd cpu_build

../generate_makefile.bash --cxxflags="-fPIC" --ldflags="-fPIC" --arch=HSW --prefix=../lib

make kokkoslib
make install

cd ..

ncpu=1
ngpu=4

for i in $(seq 0 $ncpu);
do
	cp -R cpu_build "cpu_build_${i}"
done

for i in $(seq 0 $ngpu);
do
	cp -R gpu_build "gpu_build_${i}"
done
