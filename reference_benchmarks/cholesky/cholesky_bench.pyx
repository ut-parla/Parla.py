# cython: language_level = 3

import numpy as np
import cupy as cp

cdef extern from "magma.h":
    ctypedef int magma_int_t
    magma_int_t MAGMA_SUCCESS
    magma_int_t magma_init()
    magma_int_t magma_finalize()
    enum magma_uplo_t:
        MagmaUpper
        MagmaLower
        MagmaFull
        MagmaHessenberg
    magma_int_t magma_dpotrf_mgpu(magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, double **d_IA, magma_int_t ldda, magma_int_t *info)

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
    np.random.seed(0)
    cdef int n = 16 * 4**2
    a = np.random.rand(n, n)
    a = a @ a.T
    cdef int ngpu = get_ngpu()
    local_slices = []
    step = n + (ngpu - 1) // ngpu
    cdef int i
    for i in range(ngpu):
        with cp.cuda.Device(i):
            #local_slices.append(cp.asarray(a[i::ngpu]))
            local_slices.append(cp.asarray(a[i*step:(i+1)*step]))
    cdef size_t[:] d_IA = np.empty(ngpu, np.uintp)
    for i in range(ngpu):
        d_IA[i] = local_slices[i].data.ptr
    cdef int info = 0
    cdef magma_int_t status
    status = magma_init()
    assert status == MAGMA_SUCCESS
    magma_dpotrf_mgpu(ngpu, MagmaUpper, n, <double**>&d_IA[0], n, &info)
    assert not info
    status = magma_finalize()
    assert status == MAGMA_SUCCESS
    res = np.tril(np.linalg.cholesky(a))
    for i in range(ngpu):
        #a[i::ngpu] = cp.asnumpy(local_slices[i])
        a[i*step:(i+1)*step] = cp.asnumpy(local_slices[i])
    assert np.allclose(res, np.tril(a))
