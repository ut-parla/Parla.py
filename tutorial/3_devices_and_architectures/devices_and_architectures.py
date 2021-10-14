import cupy
import numpy
from parla import Parla
from parla.array import copy, clone_here
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import spawn, TaskSpace
from parla.function_decorators import specialized


# Performs element-wise vector addition on CPUs.
@specialized
def elemwise_add():
  print("CPU kernel is called..")
  x = [1, 2, 3, 4]
  y = [5, 6, 7, 8]
  print("Output>>", *[x[i]+y[i] for i in range(len(x))], sep=' ')


# GPU variant function of elementwise_add() using variant decorator.
# This function is converted to CUDA kernel through CuPy JIT, and
# performs on GPUs.
@elemwise_add.variant(gpu)
def elemwise_add_gpu():
  print("GPU kernel is called..")
  x = cupy.array([1, 2, 3, 4])
  y = cupy.array([5, 6, 7, 8])
  cupy_elemwise_add = cupy.ElementwiseKernel("int64 x, int64 y", "int64 z", "z = x + y")
  print("Output>>", *cupy_elemwise_add(x, y), sep=' ')


def main():
  # Lesson 3-1:
  # Spawns a task on any single CPU
  # by specifying CPU architecture to placement, and
  # performs element-wise vector addition.
  @spawn(placement=cpu)
  async def start_tasks():
    @spawn(placement=cpu)
    async def cpu_arch_task():
      print("Spawns a CPU architecture task")
      elemwise_add()
    await cpu_arch_task

    # Lesson 3-2:
    # Spawns a task on any single GPU
    # by specifying GPU architecture to placement, and
    # performs elemnt-wise vector addition.
    @spawn(placement=gpu)
    async def gpu_arch_task():
      print("Spawns a GPU architecture task")
      elemwise_add()
    await gpu_arch_task

    # Lesson 3-3:
    # Spawns a task on a specific GPU, GPU0,
    # by specifying GPU0 device to placement, and
    # performs element-wise vector addition.
    @spawn(placement=gpu(0))
    async def single_gpu_task():
      print("Spawns a single GPU task")
      elemwise_add()
    await single_gpu_task

    # Lesson 3-4:
    # Spawns a task on either GPU or CPU, which is available
    # first, by specifying both GPU and CPU to placement,
    # and performs element-wise vector addition.
    @spawn(placement=[cpu, gpu])
    async def single_task_on_both():
      print("Spawns a single task on either CPU or GPU")
      elemwise_add()
    await single_task_on_both

    # Lesson 3-5:
    # Spawns a task on CPU by passing numpy array to
    # placement, and prints its values.
    arr_cpu = numpy.array([1, 2, 3, 4])
    @spawn(placement=[arr_cpu])
    async def single_task_on_cpu_with_data_placement():
      print("Spawns a single task on CPU")
      print("Specifies a placement through data location")
      print("Output>>", arr_cpu)
    await single_task_on_cpu_with_data_placement

    # Lesson 3-6:
    # Spawns a task on GPU by passing cupy array to
    # placement, and prints its values.
    arr_gpu = cupy.array([1, 2, 3, 4])
    @spawn(placement=[arr_gpu])
    async def single_task_on_gpu_with_data_placement():
      print("Spawns a single task on GPU")
      print("Specifies a placement through data location")
      print("Output>>", arr_gpu)
    await single_task_on_gpu_with_data_placement

    # Lesson 3-7:
    # Spawns four tasks on four GPU devices, respectively,
    # by specfiying GPU0 to 3 to placement, partitions
    # target operand and ouptut lists into four chunks by
    # Parla data partitioning,
    # assigns each of the chunks to each task, and
    # performs element-wise vector addition.
    NUM_GPUS=4
    # Operands for element-wise vector addition.
    x = numpy.array([1, 2, 3, 4])
    y = numpy.array([5, 6, 7, 8])
    z = numpy.array([0, 0, 0, 0])
    # Spawns each task on each GPU
    # Therefore, total four tasks are spawned and placed
    # on GPU0 to GPU3.
    for gpu_id in range(NUM_GPUS):
      @spawn(placement=gpu(gpu_id))
      async def two_gpus_task(gpu_id=gpu_id):
        # Only performs addition of the assigned chunks.
        print(f"GPU[{gpu_id}] calculates z[{gpu_id}]")
        gpu_x = clone_here(x[gpu_id:(gpu_id+1)])
        gpu_y = clone_here(y[gpu_id:(gpu_id+1)])
        z_chunk = gpu_x + gpu_y
        copy(z[gpu_id:(gpu_id+1)], z_chunk)
      await two_gpus_task
    print("Output>>", *z, sep=' ')


if __name__ == "__main__":
  with Parla():
    main()
