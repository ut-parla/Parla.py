#distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "compute.hpp" nogil:
    cdef double kokkos_function(double* array, const int N, const int device);
    cdef double kokkos_function_copy(double* array, const int N, const int dev_id);
    cdef void init(int dev);
    cdef void finalize();
    cdef void addition(float *out, float *a, float *b, const int N);
    cdef void copy_cupy(double * d_array, double* h_array, const int N, const int dev_id);
    cdef double* copy2dev(double* array, const int N, const int device);
    cdef void cleanup(double* array, const int device);

