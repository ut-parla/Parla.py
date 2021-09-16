import cupy
from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import spawn, TaskSpace
from parla.function_decorators import specialized


@specialized
def elemwise_add():
  x = [1, 2, 3]
  y = [4, 5, 6]
  print(*[x[i]+y[i] for i in range(len(x))], sep=' ')


@elemwise_add.variant(gpu)
def elemwise_add_gpu():
  x = cupy.array([1, 2, 3])
  y = cupy.array([4, 5, 6])
  cupy_elemwise_add = cupy.ElementwiseKernel("int64 x, int64 y", "int64 z", "z = x + y")
  print(*cupy_elemwise_add(x, y), sep=' ')


def main():
  arch_modes = [cpu, gpu, gpu(0)]

  for arch_mode in arch_modes:
    @spawn(placement = arch_mode)
    def elemwise_add_task():
      elemwise_add()


if __name__ == "__main__":
  with Parla():
    main()
