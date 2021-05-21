from core cimport *
import time

def cpy_copy(d_a, h_a, device):
    cdef int N = len(h_a)
    assert(len(h_a) == len(d_a))

    cdef int c_dev = device
    cdef double[:] c_ha = h_a;

    temp = <long> d_a.data.mem.ptr 
    parr = <double*> temp
    
    with nogil:
        copy_cupy(parr, &c_ha[0], N, c_dev)


def start(dev=0):
    init(dev)

def end():
    finalize()

def add_vectors(a, b, c):
    cdef int N = len(a)
    
    cdef float[:] c_a = a
    cdef float[:] c_b = b
    cdef float[:] c_out = c

    #call out of c++ & cuda code
    addition(&c_out[0], &c_a[0], &c_b[0], N)
    return c

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

def dev_copy(array, nlen, dev_id):
    t = time.perf_counter()
    cdef double result
    cdef int N = nlen
    cdef double[:] arr = array
    cdef int c_did = dev_id
    cdef unsigned long p;
    with nogil:
        p = <unsigned long> copy2dev(&arr[0], N, c_did)
    t = time.perf_counter() - t
    print("Kernel copy time", dev_id, t)
    return p


def clean(array, dev_id):
    """
    array can be host (numpy).
    If current context is a device, a copy to the device will be performed.
    """
    t = time.perf_counter()
    cdef int c_did = dev_id
    cdef unsigned long p = array;
    with nogil:
        cleanup(<double*> p, c_did)
    t = time.perf_counter() - t


def reduction(array, nlen, dev_id):
    """
    array can be host (numpy).
    If current context is a device, a copy to the device will be performed.
    """
    t = time.perf_counter()
    cdef double result
    cdef int N = nlen
    #cdef double[:] arr = array
    cdef int c_did = dev_id
    cdef unsigned long p = array;
    with nogil:
        result = kokkos_function_copy(<double*> p, N, c_did)
    t = time.perf_counter() - t
    print("Kernel mat time", dev_id, t)
    return result





