import numpy as np

from parla import Parla
from parla.cuda import gpu
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace
from parla.ldevice import LDeviceSequenceBlocked, LDeviceGridBlocked, LDeviceGridRaveled

def main():
    data = np.array([0, 0, 0, 1, 2, 3])
    print("data at the start:", data)
    mapper = LDeviceSequenceBlocked(2, placement=cpu)
    partitioned_view = mapper.partition_tensor(data)

    task = TaskSpace("Task")

    @spawn(task[0], placement=gpu)
    def t1():
        print("on GPU, data is automatically moved here as", type(partitioned_view[0]))
        print("overwrite the first half with the second half")
        partitioned_view[0] = partitioned_view[1]

    @spawn(task[1], dependencies=[task[0]], placement=cpu)
    def t2():
        print("on CPU, data is automatically moved here as", type(partitioned_view[0]))
        print("data at the end:", data)

if __name__ == '__main__':
    with Parla():
        main()
