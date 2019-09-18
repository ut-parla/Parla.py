import numpy as np

from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import *

import parla

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
parla.tasks.logger.setLevel(logging.DEBUG)
parla.cuda.logger.setLevel(logging.DEBUG)
parla.cpu.logger.setLevel(logging.DEBUG)

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

def partition_slice(i, p, n):
    return slice(i * (n // p), (i + 1) * (n // p))

def matvec_fox(y, A, x):
    """y = Ax

    Uses foxes algorithm with internal partitioning.
    """

    n, yp, Ap, xp = partition_fox(y, A, x)

    done = matvec_fox_partitioned(n, yp, Ap, xp)

    return collect_fox(n, yp, y, done)


def collect_fox(n, yp, y, done_task):
    C = TaskSpace()

    # reduce along rows
    for i in range(0, partitions_y):  # rows
        @spawn(C[i], [done_task], placement=cpu(0))
        def c():
            y[partition_slice(i, partitions_x, n)] = cpu(0).memory()(yp[i][i])

    # join the collect tasks
    @spawn(None, [C[0:partitions_y]], placement=cpu(0))
    def done():
        pass

    return done  # Return the join task


def partition_fox(y, A, x):
    # n is the size of the arrays
    n = y.shape[-1]

    # check that inputs are the correct sizes
    assert y.shape == (n,)
    assert x.shape == (n,)
    assert A.shape == (n, n)

    # FIXME: Assumes that partitions_x exactly subdivides n.
    assert n / partitions_x == n // partitions_x
    # partition A into Ap (partitions_x, partitions_y)
    Ap = [[mem(i, j)(A[partition_slice(i, partitions_x, n), partition_slice(j, partitions_y, n)])
           for j in range(partitions_x)]
          for i in range(partitions_y)]
    xp = [[mem(i, j)(x[partition_slice(j, partitions_x, n)]) if i == j else mem(i, j).np.empty(x[partition_slice(i, partitions_x, n)].shape)
           for j in range(partitions_x)]
          for i in range(partitions_y)]
    yp = [[mem(i, j).np.empty(y[partition_slice(i, partitions_x, n)].shape)
           for j in range(partitions_x)]
          for i in range(partitions_y)]
    logger.debug("Ap (placement) %r", [[v.device for v in r] for r in Ap])
    logger.debug("xp (placement) %r", [[v.device for v in r] for r in xp])
    logger.debug("yp (placement) %r", [[v.device for v in r] for r in yp])

    return n, yp, Ap, xp


def matvec_fox_partitioned(n, yp, Ap, xp):
    B = TaskSpace()
    M = TaskSpace()
    R = TaskSpace()

    # broadcast along columns
    for j in range(0, partitions_x): # columns
        for i in range(0, partitions_y): # rows
            @spawn(B[i, j], placement=loc[(i, j)])
            def b():
                xp[i][j][:] = mem(i, j)(xp[j][j])

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
            for j in range(0, partitions_x): # columns
                if i == j:
                    continue
                t = mem(i, i)(yp[i][j])
                # logger.info("%r, %r", t.device, yp[i][j].device)
                acc[:] = acc + t

    # join the reduce tasks
    @spawn(None, [R[0:partitions_y]], placement=cpu(0))
    def done():
        pass
    return done  # Return the join task

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


if __name__ == '__main__':
    @spawn(placement=cpu(0))
    async def test_fox():
        size_factor = 4*partitions_x
        A = np.random.rand(size_factor, size_factor)
        x = np.random.rand(size_factor)
        # print_actual(A, x)
        res = A @ (A @ x)
        print("----", A.shape)
        out = np.empty_like(x)
        n, yp, Ap, xp = partition_fox(out, A, x)
        T1 = matvec_fox_partitioned(n, yp, Ap, xp)
        await T1
        done = matvec_fox_partitioned(n, xp, Ap, yp)
        T = collect_fox(n, xp, out, done)
        await T
        print("++++", A.shape)
        print(np.linalg.norm(res - out, ord=np.inf))
        assert np.allclose(res, out), "Parallel fox failed"
        print("Done")
