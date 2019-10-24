#distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "kokkos_compute.hpp" nogil:
	cdef double kokkos_function(double* array, const int N, const int device);
	cdef void init(int dev);
	cdef void finalize();
