import numpy as np
from numba import jit, void, float64
import math
from scipy import linalg as la
import time

from parla.tasks import *

# Naive version of dpotrf
# Write results into lower triangle of the input array.
@jit(void(float64[:,:]), nopython=True)
def cholesky_inplace(a):
    if a.shape[0] != a.shape[1]:
        raise ValueError("A square array is required.")
    for j in range(a.shape[0]):
        a[j,j] = math.sqrt(a[j,j] - (a[j,:j] * a[j,:j]).sum())
        for i in range(j+1, a.shape[0]):
            a[i,j] -= (a[i,:j] * a[j,:j]).sum()
            a[i,j] /= a[j,j]

# This is a naive version of dtrsm.
# The result is written over the input array 'b'.
@jit(void(float64[:,:], float64[:,:]), nopython=True)
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
            @spawn(T1[j, k], [T4[j, k]])
            def t1():
                a[j,j] -= a[j,k] @ a[j,k].T
        @spawn(T2[j], [T1[j, 0:j]])
        def t2():
            time.sleep(1)
            cholesky_inplace(a[j,j])
        for i in range(j+1, a.shape[0]):
            for k in range(j):
                @spawn(T3[i, j, k], [T4[j, k], T4[i, k]])
                def t3():
                    a[i,j] -= a[i,k] @ a[j,k].T
            @spawn(T4[i, j], [T3[i, j, 0:j], T2[j]])
            def t4():
                ltriang_solve(a[j,j], a[i,j].T)
    return T2[j]

@spawn()
def test_blocked_cholesky():
    # Test all the above cholesky versions.
    a = np.random.rand(4, 4)
    a = a @ a.T
    res = la.tril(la.cho_factor(a, lower=True)[0])
    a1 = a.copy()
    cholesky_inplace(a1)
    assert np.allclose(res, la.tril(a1)), "Sequential cholesky_inplace failed"
    a1 = a.copy()
    print(a1)
    T = cholesky_blocked_inplace(a1.reshape(2,2,2,2).swapaxes(1,2))
    @spawn(None, [T])
    def t():
        print("===========")
        print(a1)
        assert np.allclose(res, la.tril(a1)), "Parallel cholesky_blocked_inplace failed"
