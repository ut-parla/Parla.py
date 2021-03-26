#distutils: language = c++

cdef extern from "gemm.h" nogil:
    cdef void internal_gemm(int m, int n, int k, const float* A, const float* B, float* C, int device);
