import cupy
from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import spawn, TaskSpace
from parla.function_decorators import specialized


@specialized
def elemwise_add(x, y):
  print("[CPU result]")
  print(*[x[i]+y[i] for i in range(len(x))], sep=' ')


@elemwise_add.variant(gpu)
def elemwise_add_gpu(x, y):
  cupy_elemwise_add = cupy.ElementwiseKernel("int64 x, int64 y", "int64 z", "z = x + y")
  print("[GPU result]")
  print(*cupy_elemwise_add(x, y), sep=' ')


def main():
  x_gpu = cupy.array([1, 2, 3])
  y_gpu = cupy.array([4, 5, 6])
  x_cpu = [1, 2, 3]
  y_cpu = [4, 5, 6]

  arch_mode = [cpu, gpu]

  for i in range(2):
    @spawn(placement = arch_mode[i])
    def elemwise_add_task():
      if (i == 0):
        elemwise_add(x_cpu, y_cpu)
      else:
        elemwise_add(x_gpu, y_gpu)


if __name__ == "__main__":
  with Parla():
    main()
