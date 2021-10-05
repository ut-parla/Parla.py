import cupy
import numpy
from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import spawn, TaskSpace
from parla.function_decorators import specialized


# Performs element-wise vector addition on CPUs.
@specialized
def elemwise_add():
  x = [1, 2, 3, 4]
  y = [5, 6, 7, 8]
  print(*[x[i]+y[i] for i in range(len(x))], sep=' ')


# GPU variant function of elementwise_add() using variant decorator.
# This function is converted to CUDA kernel through CuPy JIT, and
# performs on GPUs.
@elemwise_add.variant(gpu)
def elemwise_add_gpu():
  x = cupy.array([1, 2, 3, 4])
  y = cupy.array([5, 6, 7, 8])
  cupy_elemwise_add = cupy.ElementwiseKernel("int64 x, int64 y", "int64 z", "z = x + y")
  print(*cupy_elemwise_add(x, y), sep=' ')


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
  # Spawns four tasks on four GPU devices, respectively,
  # by specfiying GPU0 to 3 to placement, partitions
  # target operand and ouptut lists into four chunks by
  # Parla data partitioning,
  # assigns each of the chunks to each task, and
  # performs element-wise vector addition.
  NUM_GPUS=4
  # Operands for element-wise vector addition.
  x = [1, 2, 3, 4]
  y = [5, 6, 7, 8]
  c = [0, 0, 0, 0]
  # Spawns each task on each GPU.
  # Therefore, total four tasks are spawned and placed
  # on GPU0 to GPU3.
  for gpu_id in range(NUM_GPUS):
    @spawn(placement=gpu(gpu_id))
    async def two_gpus_task(gpu_id=gpu_id):
      # Only performs addition of the assigned chunks.
      print(f"GPU[{gpu_id}] calculates z[{gpu_id}]")
      z_chunk = clone_here(x[gpu_id]) + clone_here(y[gpu_id])
      copy(c[gpu_id], z_chunk)
    await two_gpus_task
  print(*z, sep=' ')

if __name__ == "__main__":
  with Parla():
    main()
