"""
A naive implementation of blocked cholesky using Numba kernels on CPUs and cuBLAS on GPUs.

This implementation does not optimize data movement *at all*. The goal is to demonstrate the simplest possible method
of transitioning from a Parla kernel implementation to an external kernel.
"""

import numpy as np
import cupy
from numba import jit, void, float64
import math
import time
import os


from parla import Parla, TaskEnvironment
from parla.array import copy, clone_here
from parla.tasks import *
from parla.cuda import *
from parla.cpu import cpu, UnboundCPUComponent
from parla.cpu import *
from parla.function_decorators import *

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
    ca = clone_here(a)
    ca[:] = cupy.linalg.cholesky(ca)
    copy(a, ca)

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
def cholesky_blocked_inplace(a):
    if a.shape[0] * a.shape[2] != a.shape[1] * a.shape[3]:
        raise ValueError("A square matrix is required.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Non-square blocks are not supported.")

    # Define task spaces
    T1 = TaskSpace("T1") # TODO: Document what each kind of task does.
    T2 = TaskSpace("T2") # Cholesky on block
    T3 = TaskSpace("T3") # TODO: ???
    T4 = TaskSpace("T4") # Triangular solve

    # TODO: Document in detail.
    for j in range(a.shape[0]):
        for k in range(j):
            @spawn(T1[j, k], [T4[j, k]])
            def t1():
                out = clone_here(a[j,j])
                rhs = clone_here(a[j,k])
                out -= rhs @ rhs.T
                copy(a[j,j], out)
        @spawn(T2[j], [T1[j, 0:j]])
        def t2():
            cholesky_inplace(a[j,j])
        for i in range(j+1, a.shape[0]):
            for k in range(j):
                @spawn(T3[i, j, k], [T4[j, k], T4[i, k]])
                def t3():
                    out = clone_here(a[i,j])
                    rhs1 = clone_here(a[i,k])
                    rhs2 = clone_here(a[j,k])
                    out -= rhs1 @ rhs2.T
                    copy(a[i,j], out)
            @spawn(T4[i, j], [T3[i, j, 0:j], T2[j]], placement=cpu(0))
            def t4():
                ltriang_solve(a[j,j], a[i,j].T)
    return T2[a.shape[0]-1]

def main():
    @spawn(placement=cpu)
    async def test_blocked_cholesky():
        # Configure environment
        n = 125 * 12
        np.random.seed(0)
        num_runs = 1
        block_size = 125
        assert not n % block_size

        # Construct input data
        a = np.random.rand(n, n)
        a = a @ a.T

        times = []
        for i in range(num_runs):
            # Copy and layout input
            a1 = a.copy()
            ap = a1.reshape(n // block_size, block_size, n // block_size, block_size).swapaxes(1,2)
            start = time.perf_counter()

            # Call Parla cholesky result and wait for completion
            await cholesky_blocked_inplace(ap)

            end = time.perf_counter()
            times.append(end - start)

        # Check result
        computed_L = np.tril(a1)
        assert(np.max(np.absolute(a - computed_L @ computed_L.T)) < 1E-8)
        print(*times)

if __name__ == '__main__':
    # Setup task execution environments
    envs = []
    envs.extend([TaskEnvironment(placement=[d], components=[GPUComponent()]) for d in gpu.devices])
    envs.extend([TaskEnvironment(placement=[d], components=[UnboundCPUComponent()]) for d in cpu.devices])
    #envs.extend([TaskEnvironment(placement=[d], components=[MultiloadComponent([CPUAffinity])]) for d in cpu.devices])
    if "N_DEVICES" in os.environ:
        envs = envs[:int(os.environ.get("N_DEVICES"))]
    # Start Parla runtime
    with Parla(envs):
        main()
