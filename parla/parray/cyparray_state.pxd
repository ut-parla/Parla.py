# distutils: language=c++

from libcpp cimport bool

cdef extern from "c/parray_state.cpp":
    pass

# a mapping between C++ PArrayState api to Cython PArrayState api
cdef extern from "c/parray_state.h" namespace "parray":
    cdef cppclass PArrayState:
        PArrayState() except +
        void set_exist_on_device(int device_id, bool exist)
        void set_valid_on_device(int device_id, bool valid)

cdef class CyPArrayState:
    cdef PArrayState *cpp_parray_state
    cdef PArrayState* get_cpp_parray_state(self)