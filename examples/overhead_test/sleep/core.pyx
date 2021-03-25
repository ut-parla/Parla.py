from sleep.core cimport sleeper, busy_sleep

def sleep(t):
    cdef unsigned int c_t = t
    with nogil:
        sleeper(c_t)

def bsleep(t):
    cdef unsigned int c_t = t
    with nogil:
        busy_sleep(c_t)
