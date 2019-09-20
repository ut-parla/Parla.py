# Run in a conda environment created like:
# conda create -n magma_test -c conda-forge magma python=3.7 scikit-build
# Be sure to also have a C++17 compatible compiler available.
python setup.py build_ext --inplace -- -DMagma_LIBRARIES="libmagma.a" -DMagma_INCLUDE_DIR="$CONDA_PREFIX/include" -DMagma_LIB_DIR="$CONDA_PREFIX/lib"
