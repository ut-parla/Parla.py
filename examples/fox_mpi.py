import itertools
import logging

import numpy as np
from mpi4py import MPI

from parla.array import copy
from parla.cpu import cpu
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
    yp, Ap, xp = partition_fox(y, A, x)

    await matvec_fox_partitioned(yp, Ap, xp)

    return await collect_fox(yp, y)


async def collect_fox(yp, y):
    C = TaskSpace()

    # Collect from diagonal
    for i in range(0, partitions_y):  # rows
        @spawn(C[i], placement=cpu(0))
        def c():
            copy(y[mapper.slice_x(i, y.shape[0])], yp[i][i])

    # join the collect tasks
    await C

    return y


def partition_fox(y, A, x):
    # # n is the size of the arrays
    # n = y.shape[0]
    #
    # # check that inputs are the correct sizes
    # assert y.shape == (n,)
    # assert x.shape == (n,)
    # assert A.shape == (n, n)

    # FIXME: Assumes that partitions_x exactly subdivides n.
    assert A.shape[0] / partitions_x == A.shape[0] // partitions_x
    assert A.shape[1] / partitions_y == A.shape[1] // partitions_y

    # partition A into Ap (partitions_x, partitions_y)
    Ap = mapper.partition_tensor(A)
    xp = mapper.partition(lambda i, j, memory:
                          x[mapper.slice_x(i, x.shape[0])]
                          if i == j else
                          memory.np.empty(x[mapper.slice_x(i, x.shape[0])].shape))
    yp = mapper.partition(lambda i, j, memory:
                          memory.np.empty(y[mapper.slice_y(j, y.shape[0])].shape))

    return yp, Ap, xp


async def matvec_fox_partitioned(yp, Ap, xp):
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


async def matvec_mpi(comm, A, x):
    m = A.shape[0] # local rows
    p = comm.Get_size()
    xg = np.zeros(m*p, dtype='d')
    comm.Allgather([x,  MPI.DOUBLE],
                   [xg, MPI.DOUBLE])
    y = np.zeros(m, dtype='d')
    return await matvec_fox(y, A, xg)


def main():
    @spawn(placement=cpu(0))
    async def test_fox():
        comm = MPI.COMM_WORLD
        print(comm.Get_rank(), comm.Get_size())

        comm.Barrier()
        size_factor = 1024*8
        A = np.random.rand(size_factor // comm.Get_size(), size_factor).astype(dtype='d')
        x = np.random.rand(size_factor // comm.Get_size()).astype(dtype='d')
        comm.Barrier()

        print("----", A.shape)
        # TODO: Doing two multiplies produces the wrong results. I suspect I've forgotten a required communication in the two mult case.
        # x = await matvec_mpi(comm, A, x)
        y = await matvec_mpi(comm, A, x)
        print("++++", A.shape)

        # TODO: The check code sometimes hangs (with processes using 100% of a core each) during the second Gather. No idea why.

        # m = A.shape[0]  # local rows
        # p = comm.Get_size()
        # yg = np.zeros(m * p, dtype='d') if comm.Get_rank() == 0 else None
        # if yg is not None: print(y.shape, yg.shape)
        # comm.Gather([y, MPI.DOUBLE],
        #             [yg, MPI.DOUBLE],
        #             root=0)
        #
        # Ag = np.zeros((m * p, size_factor), dtype='d') if comm.Get_rank() == 0 else None
        # if Ag is not None: print(A.shape, Ag.shape)
        # comm.Gather([A, mpi.DOUBLE],
        #             [Ag, mpi.DOUBLE],
        #             root=0)
        #
        # xg = np.zeros(m * p, dtype='d') if comm.Get_rank() == 0 else None
        # if xg is not None: print(x.shape, xg.shape)
        # comm.Gather([x, mpi.DOUBLE],
        #             [xg, mpi.DOUBLE],
        #             root=0)
        #
        # if comm.Get_rank() == 0:
        #     print("--", Ag.shape)
        #     res = Ag @ xg
        #     print("++", Ag.shape)
        #     print(np.linalg.norm(res - yg, ord=np.inf))
        #     assert np.allclose(res, yg), "Parallel fox failed"
        #     print("Done")



if __name__ == '__main__':
    main()
