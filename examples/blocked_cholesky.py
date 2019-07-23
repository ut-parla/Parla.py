import numpy as np
import cupy
from numba import jit, void, float64
import math
import time

from parla.tasks import *
from parla.device import *
from parla.cuda import *
from parla.cpu import *
from parla.function_decorators import *

import logging
# logging.basicConfig(level=logging.INFO)
artificial_delay = 0

# Naive version of dpotrf
# Write results into lower triangle of the input array.
@specialized
@jit(void(float64[:,:]), nopython=True, nogil=True)
def cholesky_inplace(a):
    if a.shape[0] != a.shape[1]:
        raise ValueError("A square array is required.")
    for j in range(a.shape[0]):
        a[j,j] = math.sqrt(a[j,j] - (a[j,:j] * a[j,:j]).sum())
        for i in range(j+1, a.shape[0]):
            a[i,j] -= (a[i,:j] * a[j,:j]).sum()
            a[i,j] /= a[j,j]

@cholesky_inplace.variant(gpu)
def cholesky_inplace(a):
    if a.shape[0] != a.shape[1]:
        raise ValueError("A square array is required.")
    ca = get_current_device().memory()(a) # dtype='f'
    # print("CUDA:", a, ca)
    ca[:] = cupy.linalg.cholesky(ca)
    a[:] = cpu(0).memory()(ca)
    # print("CUDA:", a, ca)

# This is a naive version of dtrsm.
# The result is written over the input array 'b'.
@jit(void(float64[:,:], float64[:,:]), nopython=True, nogil=True)
def ltriang_solve(a, b):
    if a.shape[0] != b.shape[0]:
        raise ValueError("Input array shapes are not compatible.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Array for back substitution is not square.")
    # For the implementation here, just assume lower triangular.
    for i in range(a.shape[0]):
        b[i] /= a[i,i]
        b[i+1:] -= a[i+1:,i:i+1] * b[i:i+1]

# This is a less naive version of dpotrf with one level of blocking.
# Blocks are currently assumed to evenly divide the axes lengths.
# The input array 4 dimensional. The first and second index select
# the block (row first, then column). The third and fourth index
# select the entry within the given block.
# @jit(void(float64[:,:,:,:]))
def cholesky_blocked_inplace(a):
    T1 = TaskSpace("T1")
    T2 = TaskSpace("T2")
    T3 = TaskSpace("T3")
    T4 = TaskSpace("T4")

    if a.shape[0] * a.shape[2] != a.shape[1] * a.shape[3]:
        raise ValueError("A square matrix is required.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Non-square blocks are not supported.")

    for j in range(a.shape[0]):
        # Batched BLAS operations could help here.
        for k in range(j):
            @spawn(T1[j, k], [T4[j, k]], placement=gpu(0))
            def t1():
                # print("T1", j, k)
                time.sleep(artificial_delay)
                a[j,j] -= a[j,k] @ a[j,k].T
        @spawn(T2[j], [T1[j, 0:j]], placement=gpu(0))
        def t2():
            # print("T2", j)
            time.sleep(artificial_delay)
            cholesky_inplace(a[j,j])
        for i in range(j+1, a.shape[0]):
            for k in range(j):
                @spawn(T3[i, j, k], [T4[j, k], T4[i, k]], placement=gpu(0))
                def t3():
                    # print("T3", i, j, k)
                    time.sleep(artificial_delay)
                    a[i,j] -= a[i,k] @ a[j,k].T
            @spawn(T4[i, j], [T3[i, j, 0:j], T2[j]], placement=cpu(0))
            def t4():
                # print("T4", i, j)
                time.sleep(artificial_delay)
                ltriang_solve(a[j,j], a[i,j].T)
    return T2[j]


@spawn(placement=cpu(0))
def test_blocked_cholesky():
    # Test all the above cholesky versions.
    size_factor = 15
    a = np.random.rand(16*size_factor*size_factor, 16*size_factor*size_factor)
    a = a @ a.T
    res = np.tril(np.linalg.cholesky(a))
    print("=============", a.shape)
    # a1 = a.copy()
    # cholesky_inplace(a1)
    # print(a1)
    # print("=============", a.shape)
    # assert np.allclose(res, np.tril(a1)), "Sequential cholesky_inplace failed"
    a1 = a.copy()
    # print(a1)
    # time.sleep(2)
    T = cholesky_blocked_inplace(a1.reshape(4*size_factor,4*size_factor,4*size_factor,4*size_factor).swapaxes(1,2))
    @spawn(None, [T], placement=cpu(0))
    def t():
        print("===========", a.shape)
        print(a1)
        assert np.allclose(res, np.tril(a1)), "Parallel cholesky_blocked_inplace failed"
        print("Done")
