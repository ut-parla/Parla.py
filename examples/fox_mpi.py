"""
An MPI program utilizing the Parla Fox's algorithm implemented in fox.py
"""
import numpy as np
from mpi4py import MPI

from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import *

from .fox import *


async def matvec_mpi(comm, A, x):
    m = A.shape[0] # local rows
    p = comm.Get_size()
    # Gather the value of x to all ranks
    xg = np.zeros(m*p, dtype='d')
    comm.Allgather([x,  MPI.DOUBLE],
                   [xg, MPI.DOUBLE])
    y = np.zeros(m, dtype='d')
    # Perform multiplication of A with the gathered x, xg. The rank-local result goes in y.
    return await matvec_fox(y, A, xg)


def main():
    @spawn(placement=cpu(0))
    async def test_fox():
        comm = MPI.COMM_WORLD
        print(comm.Get_rank(), comm.Get_size())

        # Create test data at each rank
        comm.Barrier()
        size_factor = 1024*8
        A = np.random.rand(size_factor // comm.Get_size(), size_factor).astype(dtype='d')
        x = np.random.rand(size_factor // comm.Get_size()).astype(dtype='d')
        comm.Barrier()

        print("----", A.shape)
        # Perform multiplication
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
