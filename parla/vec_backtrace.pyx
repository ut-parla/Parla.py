# distutils: libraries = unwind

cdef extern from "vec_backtrace.h":
    void register_handler()
    void show_backtrace_with_unwind()

register_handler()

def show_backtrace():
    show_backtrace_with_unwind()
