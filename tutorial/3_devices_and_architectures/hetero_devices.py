import cupy
from parla import Parla
from parla.array import copy, clone_here
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import spawn, TaskSpace
from parla.function_decorators import specialized


# Performs element-wise vector addition with input parameters on CPUs.
@specialized
def elemwise_add():
  print("CPU kernel is called..")
  x = [1, 2, 3, 4]
  y = [5, 6, 7, 8]
  print("Output>>", *[x[i]+y[i] for i in range(len(x))], "\n", sep=' ')


# GPU variant function of elemwise_add() using variant decorator.
@elemwise_add.variant(gpu)
def elemwise_add_gpu():
  print("GPU kernel is called..")
  x = cupy.array([1, 2, 3, 4])
  y = cupy.array([5, 6, 7, 8])
  cupy_elemwise_add = cupy.ElementwiseKernel("int64 x, int64 y", "int64 z", "z = x + y")
  print("Output>>", *cupy_elemwise_add(x, y), "\n", sep=' ')


def main():
  @spawn(placement=cpu)
  async def start_tasks():
    # Spawns a task on either GPU or CPU, which is available
    # first, by specifying both GPU and CPU to placement,
    # and performs element-wise vector addition.
    @spawn(placement=[cpu, gpu])
    async def single_task_on_both():
      print("Spawns a single task on either CPU or GPU")
      elemwise_add()
    await single_task_on_both


if __name__ == "__main__":
  with Parla():
    main()
