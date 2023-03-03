# distutils: language=c++

from .cyparray cimport PArray
from .cyparray_state cimport CyPArrayState

# a Cython wrapper class around C++ PArray
cdef class CyPArray:
    cdef PArray cpp_parray  # Hold a C++ instance which we're wrapping

    def __init__(self, uint64_t id, CyPArrayState parray_state):
        self.cpp_parray = PArray(id, parray_state.get_cpp_parray_state())

    def set_size(self, new_size):
        self.cpp_parray.set_size(new_size)