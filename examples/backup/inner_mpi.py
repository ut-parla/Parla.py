import numpy as np

from mpi4py import MPI

from parla.array import copy
from parla.cuda import gpu
from parla.cpu import cpu
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import *
import time


def main():
    comm = MPI.COMM_WORLD
    print(comm.Get_rank(), comm.Get_size())

    a = np.random.rand(10000000).astype(dtype='d')
    b = np.random.rand(10000000).astype(dtype='d')

    divisions = 100

    comm.Barrier()
    start = time.perf_counter()
    # Map the divisions onto actual hardware locations
    mapper = LDeviceSequenceBlocked(divisions)
    # print(mapper.devices)

    a_part = mapper.partition_tensor(a)
    b_part = mapper.partition_tensor(b)

    inner_result = np.empty(1, dtype='d')

    @spawn(placement=cpu(0))
    async def inner_part():
        partial_sums = np.empty(divisions)
        async with finish():
            for i in range(divisions):
                @spawn(placement=mapper.device(i))
                def inner_local():
                    copy(partial_sums[i:i+1], a_part[i] @ b_part[i])
        res = 0.
        for i in range(divisions):
            res += partial_sums[i]
        inner_result[0] = res

    overall_result = np.array(0.0, dtype='d') if comm.Get_rank() == 0 else None
    comm.Reduce([inner_result, MPI.DOUBLE],
                [overall_result, MPI.DOUBLE],
                op=MPI.SUM,
                root=0)
    if overall_result is not None:
        result = float(overall_result)
        print(result)
        end = time.perf_counter()
        print(end - start)

    assert np.allclose(np.inner(a, b), inner_result[0])

    other_results = np.empty(comm.Get_size(), dtype='d') if comm.Get_rank() == 0 else None
    comm.Gather([inner_result, MPI.DOUBLE],
                [other_results, MPI.DOUBLE],
                root=0)
    if overall_result is not None:
        assert np.isclose(result, np.sum(other_results))


if __name__ == '__main__':
    main()
