#distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "kokkos_compute.hpp" nogil:
    cdef double kokkos_function(double* array, const int N, const int device);
    cdef double kokkos_function_copy(double* array, const int N, const int dev_id);
    cdef void init(int dev);
    cdef void finalize();
    cdef void addition(float *out, float *a, float *b, const int N);

