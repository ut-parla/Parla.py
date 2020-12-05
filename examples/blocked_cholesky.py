"""
A naive implementation of blocked Cholesky using Numba kernels on CPUs.
"""

import numpy as np
# import cupy
from numba import jit, void, float64
import math
import time

from parla import Parla
from parla.array import copy, clone_here
from parla.tasks import *
# from parla.cuda import *
from parla.cpu import *


@jit(void(float64[:,:]), nopython=True, nogil=True)
def cholesky_inplace(a):
    """
    Naive version of dpotrf. Write results into lower triangle of the input array.
    """
    if a.shape[0] != a.shape[1]:
        raise ValueError("A square array is required.")
    for j in range(a.shape[0]):
        a[j,j] = math.sqrt(a[j,j] - (a[j,:j] * a[j,:j]).sum())
        for i in range(j+1, a.shape[0]):
            a[i,j] -= (a[i,:j] * a[j,:j]).sum()
            a[i,j] /= a[j,j]


@jit(void(float64[:,:], float64[:,:]), nopython=True, nogil=True)
def ltriang_solve(a, b):
    """
    This is a naive version of dtrsm. The result is written over the input array `b`.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("Input array shapes are not compatible.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Array for back substitution is not square.")
    # For the implementation here, just assume lower triangular.
    for i in range(a.shape[0]):
        b[i] /= a[i,i]
        b[i+1:] -= a[i+1:,i:i+1] * b[i:i+1]


def cholesky_blocked_inplace(a):
    """
    This is a less naive version of dpotrf with one level of blocking.
    Blocks are currently assumed to evenly divide the axes lengths.
    The input array 4 dimensional. The first and second index select
    the block (row first, then column). The third and fourth index
    select the entry within the given block.
    """
    if a.shape[0] * a.shape[2] != a.shape[1] * a.shape[3]:
        raise ValueError("A square matrix is required.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Non-square blocks are not supported.")

    # Define task spaces
    gemm1 = TaskSpace("gemm1")        # Inter-block GEMM
    subcholesky = TaskSpace("subcholesky")  # Cholesky on block
    gemm2 = TaskSpace("gemm2")        # Inter-block GEMM
    solve = TaskSpace("solve")        # Triangular solve

    for j in range(a.shape[0]):
        for k in range(j):
            # Inter-block GEMM
            @spawn(gemm1[j, k], [solve[j, k]])
            def t1():
                out = a[j,j]
                rhs = a[j,k]

                out -= rhs @ rhs.T

                a[j,j] = out

        # Cholesky on block
        @spawn(subcholesky[j], [gemm1[j, 0:j]])
        def t2():
            cholesky_inplace(a[j,j])

        for i in range(j+1, a.shape[0]):
            for k in range(j):
                # Inter-block GEMM
                @spawn(gemm2[i, j, k], [solve[j, k], solve[i, k]])
                def t3():
                    out = a[i,j]
                    rhs1 = a[i,k]
                    rhs2 = a[j,k]

                    out -= rhs1 @ rhs2.T

                    a[i,j] = out

            # Triangular solve
            @spawn(solve[i, j], [gemm2[i, j, 0:j], subcholesky[j]])
            def t4():
                ltriang_solve(a[j,j], a[i,j].T)

    return subcholesky[a.shape[0]-1]

def main():
    @spawn(placement=cpu)
    async def test_blocked_cholesky():
        # Configure environment
        n = 125 * 16
        block_size = 125
        assert not n % block_size

        # Construct input data
        a = np.random.rand(n, n)
        a = a @ a.T

        # Copy and layout input
        a1 = a.copy()
        ap = a1.reshape(n // block_size, block_size, n // block_size, block_size).swapaxes(1,2)
        start = time.perf_counter()

        # Call Parla Cholesky result and wait for completion
        await cholesky_blocked_inplace(ap)

        end = time.perf_counter()
        print(end - start, "seconds")

        # Check result
        computed_L = np.tril(a1)
        assert(np.max(np.absolute(a - computed_L @ computed_L.T)) < 1E-8)

if __name__ == '__main__':
    with Parla():
        main()
