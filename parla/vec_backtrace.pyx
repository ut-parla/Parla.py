# distutils: libraries = unwind

cdef extern from "vec_backtrace.h":
    void register_handler()

register_handler()
