# cython: language_level = 3

import numpy as np
import cupy as cp
import time

from libc.stdlib cimport malloc, free

cdef extern from "magma_auxiliary.h" nogil:
    cppclass magma_queue_t:
        pass
    void magma_queue_create(int device, magma_queue_t *queue_ptr)
    void magma_queue_destroy(magma_queue_t queue_ptr)

cdef extern from "magma_v2.h" nogil:
    ctypedef int magma_int_t
    magma_int_t MAGMA_SUCCESS
    magma_int_t magma_init()
    magma_int_t magma_finalize()
    void magma_dsetmatrix_1D_col_bcyclic(magma_int_t ngpu, magma_int_t m, magma_int_t n, magma_int_t nb, double *hA, magma_int_t lda, double **dA, magma_int_t ldda, magma_queue_t *queues)
    void magma_dgetmatrix_1D_col_bcyclic(magma_int_t ngpu, magma_int_t m, magma_int_t n, magma_int_t nb, double **da, magma_int_t ldda, double *hA, magma_int_t lda, magma_queue_t *queues)
    enum magma_uplo_t:
        MagmaUpper
        MagmaLower
        MagmaFull
        MagmaHessenberg
    magma_int_t magma_dpotrf_mgpu(magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, double **d_IA, magma_int_t ldda, magma_int_t *info)
    magma_int_t magma_get_dpotrf_nb(magma_int_t n)

# Hack to get number of GPUs from CuPy.
cdef int get_ngpu():
    cdef int ngpu = 0
    while True:
        next_device = cp.cuda.Device(ngpu)
        try:
            next_device.compute_capability
        except cp.cuda.runtime.CUDARuntimeError:
            break
        ngpu += 1
    return ngpu

def bench_cholesky():
    num_runs = 50
    np.random.seed(0)
    cdef int n = 6000
    cdef magma_int_t status
    status = magma_init()
    assert status == MAGMA_SUCCESS
    cdef int nb = magma_get_dpotrf_nb(n)
    data = np.random.rand(n, n)
    cdef double[:,:] a = data @ data.T
    cdef double[:,:] a_c = np.copy(a)
    # Reference result to compare against.
    res = np.tril(np.linalg.cholesky(a))
    cdef int ngpu = get_ngpu()
    local_buffers = []
    num_blocks = (n + nb - 1) // nb
    max_blocks_per_device = (num_blocks + ngpu - 1) // ngpu
    local_buffer_size = n * nb * max_blocks_per_device
    cdef int i
    for i in range(ngpu):
        with cp.cuda.Device(i):
            local_buffers.append(cp.empty(local_buffer_size, 'd'))
    cdef size_t[:] d_IA = np.empty(ngpu, np.uintp)
    for i in range(ngpu):
        d_IA[i] = local_buffers[i].data.ptr
    cdef int info = 0
    cdef magma_queue_t *queues = <magma_queue_t*>malloc(ngpu * sizeof(magma_queue_t))
    for i in range(ngpu):
        magma_queue_create(i, &queues[i])
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        magma_dsetmatrix_1D_col_bcyclic(ngpu, n, n, nb, &a[0,0], n, <double**>&d_IA[0], n, &queues[0])
        magma_dpotrf_mgpu(ngpu, MagmaUpper, n, <double**>&d_IA[0], n, &info)
        assert not info, info
        magma_dgetmatrix_1D_col_bcyclic(ngpu, n, n, nb, <double**>&d_IA[0], n, &a_c[0,0], n, &queues[0])
        end = time.perf_counter()
        times.append(end - start)
    for i in range(ngpu):
        magma_queue_destroy(queues[i])
    free(<void*>queues)
    status = magma_finalize()
    assert status == MAGMA_SUCCESS
    assert np.allclose(res, np.tril(a_c))
    print(*times)
