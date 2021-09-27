import cupy
from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import spawn, TaskSpace
from parla.function_decorators import specialized
from parla.ldevice import LDeviceSequenceBlocked


@specialized
def elemwise_add():
  x = [1, 2, 3, 4]
  y = [5, 6, 7, 8]
  print(*[x[i]+y[i] for i in range(len(x))], sep=' ')


@elemwise_add.variant(gpu)
def elemwise_add_gpu():
  x = cupy.array([1, 2, 3, 4])
  y = cupy.array([5, 6, 7, 8])
  cupy_elemwise_add = cupy.ElementwiseKernel("int64 x, int64 y", "int64 z", "z = x + y")
  print(*cupy_elemwise_add(x, y), sep=' ')


def main():
  @spawn(placement=cpu)
  async def start_tasks():
    cpu_arch_task_space = TaskSpace("cpu_arch_task")
    @spawn(cpu_arch_task_space, placement=cpu)
    def cpu_arch_task():
      print("Spawns a CPU architecture task")
      elemwise_add()

    gpu_arch_task_space = TaskSpace("gpu_arch_task")
    @spawn(gpu_arch_task_space, placement=gpu, dependencies=[cpu_arch_task_space])
    def gpu_arch_task():
      print("Spawns a GPU architecture task")
      elemwise_add()

    single_gpu_task_space = TaskSpace("single_gpu_task")
    @spawn(single_gpu_task_space, placement=gpu(0), dependencies=[gpu_arch_task_space])
    def single_gpu_task():
      print("Spawns a single GPU task")
      elemwise_add()
    await single_gpu_task_space

    multi_gpus_task_space = TaskSpace("multi_gpus_task")
    NUM_GPUS=2
    x = cupy.array([1, 2, 3, 4])
    y = cupy.array([5, 6, 7, 8])
    mapper = LDeviceSequenceBlocked(2, placement=gpu)
    x_view = mapper.partition_tensor(x)
    y_view = mapper.partition_tensor(y)
    print(x_view[0], " --> x0")
    print(x_view[1], " --> x1")
    print(y_view[0], " --> y0")
    print(y_view[1], " --> y1")

#    for gpu_id in range(NUM_GPUS):
#      @spawn(task[3+gpu_id], placement=gpu(gpu_id))
#      def two_gpus_task(gpu_id=gpu_id):
#        pass
#    await task[2+NUM_GPUS]


if __name__ == "__main__":
  with Parla():
    main()
