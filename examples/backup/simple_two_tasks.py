from parla import Parla
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace

def different_taskspace_dependency():
  first_task = TaskSpace("FirstTask")
  @spawn(first_task, memory=2000)
  def t0():
    print("\tTask[0]")

  # Spawns a task on the different task space
  # from the first task which is dependent on
  # the first task.
  second_task = TaskSpace("SecondTask")
  @spawn(second_task, [first_task])
  def t1():
    print("\tTask[1]")
  # Return the last task.
  return second_task

if __name__ == "__main__":
  with Parla():
    print("Two simple tasks started")
    different_taskspace_dependency()
