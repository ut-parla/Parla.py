# distutils: language=c++

from libc.stdint cimport uint64_t

from .cyparray_state cimport PArrayState

cdef extern from "c/parray.cpp":
    pass

# a mapping between C++ PArray api to Cython PArray api
cdef extern from "c/parray.h" namespace "parray":
    cdef cppclass PArray:
        PArray() except +
        PArray(uint64_t, PArrayState *) except +
        void set_size(uint64_t)
