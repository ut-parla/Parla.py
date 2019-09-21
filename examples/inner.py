import numpy as np
from parla.cuda import gpu
from parla.cpu import cpu
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import *


def main():
    a = np.random.rand(10000000)
    b = np.random.rand(10000000)

    divisions = 100

    # Map the divisions onto actual hardware locations
    mapper = LDeviceSequenceBlocked(divisions)
    print(mapper)

    a_part = mapper.partition_tensor(a)
    b_part = mapper.partition_tensor(b)
    # print(a)
    # print(a_part)
    # print(b)
    # print(b_part)

    inner_result = np.empty(1)

    @spawn(placement = cpu(0))
    async def inner_part():
        divisions = len(a_part)
        partial_sums = np.empty(divisions)
        async with finish():
            for i in range(divisions):
                @spawn(placement = mapper.device(i))
                def inner_local():
                    partial_sums[i] = float(a_part[i] @ b_part[i])
        res = 0.
        for i in range(divisions):
            res += partial_sums[i]
        inner_result[0] = res

    # print(np.inner(a, b), inner_result[0])
    assert np.allclose(np.inner(a, b), inner_result[0])


if __name__ == '__main__':
    main()
