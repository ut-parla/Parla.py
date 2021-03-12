#distutils: language = c++

cdef extern from "sleep.h" nogil:
    cdef void sleeper(int t);
    cdef void busy_sleep(int milli);
