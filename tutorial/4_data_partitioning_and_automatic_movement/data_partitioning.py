import numpy as np

from parla import Parla
from parla.cuda import gpu
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace
from parla.ldevice import LDeviceSequenceBlocked, LDeviceGridBlocked, LDeviceGridRaveled

def main():
    data = np.random.rand(4)
    print("Initial array: ", data)

    print("=======")

    mapper = LDeviceSequenceBlocked(2, placement=[cpu[0], gpu[0]])
    partitioned_view = mapper.partition_tensor(data)
    print("Partitions:")
    print("\n".join(f"{v} of type {t} on device {d}" for v,t,d in zip(partitioned_view.base, partitioned_view.types, partitioned_view.devices)))

    print("=======")
    task = TaskSpace("Task")

    @spawn(task[0], placement=gpu[0])
    def t1():
        sum_on_gpu = partitioned_view[0] + partitioned_view[1]
        print("On GPU, inputs are automatically cloned here")
        print("input types:", [type(i) for i in partitioned_view])
        print("output type:", type(sum_on_gpu))
        print("output:", sum_on_gpu)
        print("=======")

    @spawn(task[1], dependencies=[task[0]], placement=cpu[0])
    def t2():
        sum_on_cpu = partitioned_view[0] + partitioned_view[1]
        print("On CPU, inputs are automatically cloned here")
        print("input types:", [type(i) for i in partitioned_view])
        print("output type:", type(sum_on_cpu))
        print("output:", sum_on_cpu)

if __name__ == '__main__':
    with Parla():
        main()
