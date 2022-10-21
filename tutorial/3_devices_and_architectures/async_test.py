import cupy
import numpy
from parla import Parla
from parla.array import copy, clone_here
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import spawn, TaskSpace
from parla.function_decorators import specialized

import numpy as np

# Performs element-wise vector addition on CPUs.
@specialized
def elemwise_add():
  print("CPU kernel is called..")
  x = [1, 2, 3, 4]
  y = [5, 6, 7, 8]
  print("Output>>", *[x[i]+y[i] for i in range(len(x))], sep=' ')


# GPU variant function of elemwise_add() using variant decorator.
# This function is converted to CUDA kernel through CuPy JIT, and
# performs on GPUs.
@elemwise_add.variant(gpu)
def elemwise_add_gpu():
  print("GPU kernel is called..")
  x = cupy.array([1, 2, 3, 4])
  y = cupy.array([5, 6, 7, 8])
  cupy_elemwise_add = cupy.ElementwiseKernel("int64 x, int64 y", "int64 z", "z = x + y")
  print("Output>>", *cupy_elemwise_add(x, y), sep=' ')


# Performs element-wise vector addition with input parameters on CPUs.
@specialized
def elemwise_add_with_params(x, y):
  print("CPU kernel is called..")
  print("Output>>", *[x[i]+y[i] for i in range(len(x))], sep=' ')


# GPU variant function of elemwise_add_with_params() using
# variant decorator.
@elemwise_add_with_params.variant(gpu)
def elemwise_add_with_params_gpu(x, y):
  print("GPU kernel is called..", flush=True)
  cupy_elemwise_add = cupy.ElementwiseKernel("int64 x, int64 y", "int64 z", "z = x + y")
  print("Output>>", *cupy_elemwise_add(x, y), sep=' ')


def main():

  taskspace = TaskSpace("ts")

  @spawn(placement=cpu)
  async def start_tasks():

    print("Spawning", flush=True)
    @spawn(taskspace[0], placement=gpu)
    async def gpu_arch_task():
      print("Spawns a GPU architecture task", flush=True)
      for i in range(5):
          elemwise_add()
          A = np.random.rand(100, 100)
          B = np.random.rand(100, 100)
          C = A @ B.T


    await gpu_arch_task
    print("HERE", flush=True)

if __name__ == "__main__":
  with Parla():
    main()
