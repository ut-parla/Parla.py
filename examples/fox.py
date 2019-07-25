from typing import List

import numpy as np

from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import *

import logging
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
This is not actually runnable. Think of it as very detailed pseudocode.
We will make it run soon-ish. Both library changes and code changes here will be needed.
"""

loc = {
    (0, 0): gpu(0),
    (0, 1): gpu(0),
    (1, 0): gpu(0),
    (1, 1): gpu(0),
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

    # FIXME: Assumes that partitions_x exactly subdivides n.
    # partition A into Ap (partitions_x, partitions_y)
    Ap = [[mem(i, j)(A[partition_slice(i, partitions_x), partition_slice(j, partitions_y)])
            for j in range(partitions_x)]
          for i in range(partitions_y)]

    xp = [[mem(i, j).np.empty(x[partition_slice(j, partitions_x)].shape)
              for j in range(partitions_x)]
          for i in range(partitions_y)]

    yp = [[mem(i, j).np.empty(x[partition_slice(i, partitions_x)].shape)
              for j in range(partitions_x)]
          for i in range(partitions_y)]

    logger.debug("Ap (placement) %r", [[v.device for v in r] for r in Ap])
    logger.debug("xp (placement) %r", [[v.device for v in r] for r in xp])
    logger.debug("yp (placement) %r", [[v.device for v in r] for r in yp])

    # broadcast along columns
    for j in range(0, partitions_x): # columns
        for i in range(0, partitions_y): # rows
            @spawn(B[i, j], placement=loc[(i, j)])
            def b():
                xp[i][j][:] = mem(i, j)(x[partition_slice(j, partitions_x)])

    # block-wise multiplication
    for i in range(0, partitions_y):  # rows
        for j in range(0, partitions_x): # columns
            @spawn(M[i, j], [B[i, j]], placement=loc[(i, j)])
            def m():
                # TODO: Once cupy supports the out parameter for matmul, use that here instead.
                yp[i][j][:] = Ap[i][j] @ xp[i][j]

    # reduce along rows
    for i in range(0, partitions_y): # rows
        @spawn(R[i], [M[i, 0:partitions_x]], placement=loc[(i, i)])
        def r():
            acc = yp[i][i]
            # logger.info("acc = %r (at %r)", acc.device, get_current_device())
            for j in range(0, partitions_y): # columns
                if i == j:
                    continue
                t = mem(i, i)(yp[i][j])
                # logger.info("%r, %r", t.device, yp[i][j].device)
                acc[:] = acc + t
            y[partition_slice(i, partitions_x)] = cpu(0).memory()(acc)

    # join the reduce tasks
    @spawn(None, [R[0:partitions_x]], placement=cpu(0))
    def done():
        pass
    return done # Return the join task

def print_actual(A, x):
    assert A.ndim == 2
    assert x.ndim == 1
    assert A.shape[0] == A.shape[1] == x.shape[0]
    y = np.zeros_like(x)
    dim_size = x.shape[0]
    block_size_x = dim_size // partitions_x
    block_size_y = dim_size // partitions_y
    contributions = []
    for i in range(partitions_y):
        contributions.append([])
        for j in range(partitions_x):
            actual_contribution = A[i*block_size_y:(i+1)*block_size_y,j*block_size_x:(j+1)*block_size_x] @ x[j*block_size_x:(j+1)*block_size_x]
            print("actual ({}, {}) A input block".format(i, j))
            print(A[i*block_size_y:(i+1)*block_size_y,j*block_size_x:(j+1)*block_size_x])
            print("actual ({}, {}) x input block".format(i, j))
            print(x[j*block_size_x:(j+1)*block_size_x])
            print("actual ({}, {}) output block:".format(i, j))
            print(actual_contribution)
            contributions[-1].append(actual_contribution)
    for i in range(partitions_y):
        y_slice = y[i*block_size_y:(i+1)*block_size_y]
        for j in range(partitions_x):
            y_slice += contributions[i][j]
        print("actual contribution {}:".format(i))
        print(y_slice)
    assert np.allclose(A @ x, y)

@spawn(placement=cpu(0))
def test_fox():
    size_factor = 2*partitions_x
    A = np.random.rand(size_factor, size_factor)
    x = np.random.rand(size_factor)
    #print_actual(A, x)
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
        print(res)
        print(out)
        print(A1)
        print(x)
        assert np.allclose(res, out), "Parallel fox failed"
        print("Done")
