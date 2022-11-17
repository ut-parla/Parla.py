import cupy
import numpy
from parla import Parla
from parla.array import copy, clone_here
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import spawn, TaskSpace


# GPU variant function of elemwise_add_with_params() using
# variant decorator.
def elemwise_add(x, y):
  print("GPU kernel is called..")
  cupy_elemwise_add = cupy.ElementwiseKernel("int64 x, int64 y", "int64 z", "z = x + y")
  return cupy_elemwise_add(x, y)


def main():
  @spawn(placement=cpu)
  async def start_tasks():
    # Spawns a task on GPU by passing cupy array to
    # placement, and prints its values.
    @spawn(placement=[gpu])
    async def gpuarch_place_case():
      # Operands for element-wise vector addition.
      x_g = cupy.array([1, 2, 3, 4])
      y_g = cupy.array([5, 6, 7, 8])
      print("This should be running on GPU")
      z_g = elemwise_add(x_g, y_g)
      print("Output>>", z_g, "\n")
    await gpuarch_place_case 


    # Spawns four tasks on four GPU devices, respectively,
    # by specfiying GPU0 to 3 to placement, partitions
    # target operand and ouptut lists into four chunks by
    # Parla data partitioning,
    # assigns each of the chunks to each task, and
    # performs element-wise vector addition.
    NUM_GPUS=2
    # Operands for element-wise vector addition.
    x_c = numpy.array([1, 2, 3, 4])
    y_c = numpy.array([5, 6, 7, 8])
    z_c = numpy.array([0, 0, 0, 0])
    # Spawns each task on each GPU
    # Therefore, total four tasks are spawned and placed
    # on GPU0 to GPU3.
    for gpu_id in range(NUM_GPUS):
      @spawn(placement=gpu(gpu_id))
      async def workpart_across_gpus(gpu_id=gpu_id):
        # Only performs addition of the assigned chunks.
        print(f"GPU[{gpu_id}] calculates z[{gpu_id}]")
        tmp_x = clone_here(x_c[gpu_id:(gpu_id+1)])
        tmp_y = clone_here(y_c[gpu_id:(gpu_id+1)])
        z_chunk = elemwise_add(tmp_x, tmp_y)
        copy(z_c[gpu_id:(gpu_id+1)], z_chunk)
      await workpart_across_gpus 
    print("Output>>", *z_c, sep=' ')


if __name__ == "__main__":
  with Parla():
    main()
