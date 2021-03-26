from core cimport internal_gemm
cimport cython 

@cython.boundscheck(False)
@cython.wraparound(False)
def gemm(A, B, C, dev):
    cdef int m = A.shape[0]
    cdef int n = B.shape[1]
    cdef int k = A.shape[1]
    cdef int c_dev = dev

    assert(k == B.shape[0])

    tempA = <long> A.data.mem.ptr
    cdef float* pA = <float*> tempA

    tempB = <long> B.data.mem.ptr
    cdef float* pB = <float*> tempB

    tempC = <long> C.data.mem.ptr
    cdef float* pC = <float*> tempC

    with nogil:
        internal_gemm(m, n, k, pA, pB, pC, c_dev);
