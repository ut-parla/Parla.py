#!/bin/bash


cd "${KOKKOS_DIR}"

rm -rf gpu_build
mkdir gpu_build
cd gpu_build

CC="$KOKKOS_DIR/bin/nvcc_wrapper" CXX="$KOKKOS_DIR/bin/nvcc_wrapper" ../generate_makefile.bash --cxxflags="-fPIC" --ldflags="-fPIC" --arch=Pascal60 --prefix=${KOKKOS_DIR}/gpu_build/lib/ --with-cuda --with-cuda-options="enable_lambda" --disable-tests

make -j 10 kokkoslib
make install


cd ..

rm -rf cpu_build
mkdir cpu_build
cd cpu_build

../generate_makefile.bash --cxxflags="-fPIC -fopenmp" --ldflags="-fPIC" --arch=HSW --prefix=${KOKKOS_DIR}/cpu_build/lib/ --disable-tests

make -j 10 kokkoslib
make install

