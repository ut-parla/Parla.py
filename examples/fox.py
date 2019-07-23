from typing import List

import numpy as np

from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import *

import logging
logging.basicConfig(level=logging.DEBUG)

"""
This is not actually runnable. Think of it as very detailed pseudocode.
We will make it run soon-ish. Both library changes and code changes here will be needed.
"""

loc = {
    (0, 0): gpu(0),
    (0, 1): gpu(1),
    (1, 0): gpu(2),
    (1, 1): gpu(3),
    }
# Eventually: loc = infer_placements()
# Or even better: loc = abstract_cartesian(n, n)
def mem(i, j):
    return loc[(i,j)].memory()

partitions_x = 2
partitions_y = partitions_x

# TODO: Compare to how this is done in HPF (High-Performance Fortran).
#  Alignment: an abstract set of storage locations and then placement of data in those abstract locations.
#  Distribution (at runtime): associate abstract locations to physical locations. This mapping was static.
#
# (Reference Pingali 1989)

# Examples:
#  Fox's Algorithm: Collective regular communication (broadcast and reduce)
#  Cholesky: Less regular data movement and task placement, more complex dependencies.


def fox(y, A, x):
    """y = Ax

    Uses foxes algorithm with internal partitioning.
    """

    B = TaskSpace()
    M = TaskSpace()
    R = TaskSpace()

    # n is the size of the arrays
    n = y.shape[-1]

    # check that inputs are the correct sizes
    assert y.shape == (n,)
    assert x.shape == (n,)
    assert A.shape == (n, n)

    def partition_slice(i, p):
        return slice(i*(n//p),(i+1)*(n//p))

    # FIXME: Assume that partitions_x exactly subdivides n.
    # partition A into Ap (partitions_x, partitions_y)
    Ap: List[List[np.ndarray]]
    Ap = [[mem(i, j)(A[partition_slice(i, partitions_x), partition_slice(j, partitions_y)])
            for j in range(partitions_x)]
          for i in range(partitions_y)]

    xp = [[mem(i, j).np.empty(x[partition_slice(i, partitions_x)].shape)
              for j in range(partitions_x)]
          for i in range(partitions_y)]

    yp = [[mem(i, j).np.empty(x[partition_slice(i, partitions_x)].shape)
              for j in range(partitions_x)]
          for i in range(partitions_y)]

    # broadcast along columns
    for j in range(0, partitions_x): # columns
        for i in range(0, partitions_y): # rows
            @spawn(B[i, j], placement=loc[(i, j)])
            def b():
                xp[i][j] = mem(i, j)(x[partition_slice(i, partitions_x)])

    # block-wise multiplication
    for i in range(0, partitions_y):  # rows
        for j in range(0, partitions_x): # columns
            @spawn(M[i, j], [B[i, j]], placement=loc[(i, j)])
            def m():
                yp[i][j] = Ap[i][j] @ xp[i][j]

    # reduce along rows
    for i in range(0, partitions_x): # rows
        @spawn(R[i], [M[i, 0:partitions_x]], placement=loc[(i, i)])
        def r():
            acc = yp[i][i]
            for j in range(0, partitions_y): # columns
                if i == j:
                    continue
                t = mem(i, i)(yp[i][j])
                acc = acc + t
            y[partition_slice(i, partitions_x)] = cpu(0).memory()(acc)

    # join the reduce tasks
    @spawn(None, [R[0:partitions_x]], placement=cpu(0))
    def done():
        pass
    return done # Return the join task



@spawn(placement=cpu(0))
def test_fox():
    size_factor = 4*partitions_x
    A = np.random.rand(size_factor, size_factor)
    x = np.random.rand(size_factor)
    res = A @ x
    print("=============", A.shape)
    print(res)
    print(A)
    print(x)
    A1 = A.copy()
    # print(a1)
    # time.sleep(2)
    out = np.empty_like(res)
    T = fox(out, A, x)
    @spawn(None, [T], placement=cpu(0))
    def check():
        print("===========", A.shape)
        print(out)
        print(A1)
        print(x)
        assert np.allclose(res, out), "Parallel fox failed"
        print("Done")
