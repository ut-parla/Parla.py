import itertools
import logging

import numpy as np

import parla
from parla.array import copy
from parla.cpu import cpu
from parla.cuda import gpu
from parla.device import get_all_devices
from parla.ldevice import LDeviceGridBlocked
from parla.tasks import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# parla.tasks.logger.setLevel(logging.DEBUG)
# parla.cuda.logger.setLevel(logging.DEBUG)
# parla._cpuutils.logger.setLevel(logging.DEBUG)

partitions_x = 8
partitions_y = partitions_x

mapper = LDeviceGridBlocked(partitions_x, partitions_y)
print(mapper)

async def matvec_fox(y, A, x):
    """y = Ax

    Uses foxes algorithm with internal partitioning.
    """
    n, yp, Ap, xp = partition_fox(y, A, x)

    await matvec_fox_partitioned(n, yp, Ap, xp)

    return await collect_fox(n, yp, y)


async def collect_fox(n, yp, y):
    C = TaskSpace()

    # Collect from diagonal
    for i in range(0, partitions_y):  # rows
        @spawn(C[i], placement=cpu(0))
        def c():
            copy(y[mapper.slice_x(i, n)], yp[i][i])

    # join the collect tasks
    await C

    return y


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
    Ap = mapper.partition_tensor(A)
    xp = mapper.partition(lambda i, j, memory:
                          x[mapper.slice_x(i, n)]
                          if i == j else
                          memory.np.empty(x[mapper.slice_x(i, n)].shape))
    yp = mapper.partition(lambda i, j, memory:
                          memory.np.empty(y[mapper.slice_y(j, n)].shape))

    return n, yp, Ap, xp


async def matvec_fox_partitioned(n, yp, Ap, xp):
    B = TaskSpace()
    M = TaskSpace()
    R = TaskSpace()

    # broadcast along columns
    for j, i in itertools.product(range(0, partitions_x), range(0, partitions_y)):
        @spawn(B[i, j], placement=mapper.device(i, j))
        def b():
            xp[i][j][:] = mapper.memory(i, j)(xp[j][j])

    # block-wise multiplication
    for i, j in itertools.product(range(0, partitions_y), range(0, partitions_x)):
        @spawn(M[i, j], [B[i, j]], placement=mapper.device(i, j))
        def m():
            # TODO: Once cupy supports the out parameter for matmul, use that here instead.
            yp[i][j][:] = Ap[i][j] @ xp[i][j]

    # reduce along rows
    for i in range(0, partitions_y):  # rows
        @spawn(R[i], [M[i, 0:partitions_x]], placement=mapper.device(i, i))
        def r():
            acc = yp[i][i]
            for j in range(0, partitions_x):  # columns
                if i == j:
                    continue
                acc[:] = acc + mapper.memory(i, i)(yp[i][j])

    # join the reduce tasks
    await R


def main():
    @spawn(placement=cpu(0))
    async def test_fox():
        size_factor = 1024
        A = np.random.rand(size_factor, size_factor)
        x = np.random.rand(size_factor)

        res = A @ x
        print("----", A.shape)
        out = np.empty_like(x)
        out1 = await matvec_fox(out, A, x)
        assert out is out1
        print("++++", A.shape)
        print(np.linalg.norm(res - out, ord=np.inf))
        assert np.allclose(res, out), "Parallel fox failed"

        res = A @ (A @ x)
        print("----", A.shape)
        out = np.empty_like(x)
        n, yp, Ap, xp = partition_fox(out, A, x)
        await matvec_fox_partitioned(n, yp, Ap, xp)
        await matvec_fox_partitioned(n, xp, Ap, yp)
        out1 = await collect_fox(n, xp, out)
        assert out is out1
        print("++++", A.shape)
        print(np.linalg.norm(res - out, ord=np.inf))
        assert np.allclose(res, out), "Parallel fox failed"
        print("Done")


if __name__ == '__main__':
    main()
