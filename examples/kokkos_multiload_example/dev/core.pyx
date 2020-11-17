from core cimport *

def start(dev=0):
    init(dev)

def end():
    finalize()

"""
def reduction(array):
    cdef int N = len(array)
    cdef long device = -1
    cdef double result
    cdef double[:] arr
    
    cdef long temp
    cdef double* parr
        arr = array
        result = kokkos_function(&arr[0], N, device)
    else: #assume CuPy array 
        temp = <long> array.data.mem.ptr
        device = <long> array.data.device.id
        parr = <double *> temp
        result = kokkos_function(parr, N, device)
    return result
"""

def reduction(array):
    """
    array can be host (numpy).
    If current context is a device, a copy to the device will be performed.
    """
    result = None
    cdef int N = len(array)
    cdef double[:] arr

    arr = array
    result = kokkos_function_copy(&arr[0], N)

    return result





