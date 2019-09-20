# Implementation in progress. Do this after building and linking with magma works.

import numpy as np
import cupy as cp

cdef extern from "magma_d.h":
    ctypedef int magma_int_t
    magma_int_t magma_init()
    magma_int_t magma_finalize()
    enum magma_uplo_t:
        MagmaUpper
        MagmaLower
        MagmaFull
        MagmaHessenberg
    magma_int_t magma_dpotrf_mgpu(magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, double **d_IA, magma_int_t ldda, magma_int_t *info)


