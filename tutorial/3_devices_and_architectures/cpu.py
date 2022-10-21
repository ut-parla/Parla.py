import numpy
from parla import Parla
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace


# Performs element-wise vector addition with input parameters on CPUs.
def elemwise_add(x, y):
  print("CPU kernel is called..")
  print("Output>>", *[x[i]+y[i] for i in range(len(x))], "\n", sep=' ')


def main():
  @spawn(placement=cpu)
  async def start_tasks():
    x = numpy.array([1, 2, 3, 4])
    y = numpy.array([5, 6, 7, 8])

    # Spawns a task on any single CPU core
    # by specifying CPU architecture to placement, and
    # performs element-wise vector addition.
    @spawn(placement=cpu)
    async def single_cpu_task():
      print("[Lesson 3-1] Spawns a CPU architecture task.")
      elemwise_add(x, y)
    await single_cpu_task 

    # Spawns a task on CPU by passing numpy array to
    # placement, and prints its values.
    @spawn(placement=[x, y])
    async def single_cpu_task_through_data_placement():
      print("[Lesson 3-1] Specifies a placement through data location")
      print("This should be running on CPU")
      print("Spawns a single task on CPU")
      elemwise_add(x, y)
    await single_cpu_task_through_data_placement


if __name__ == "__main__":
  with Parla():
    main()
