from parla import Parla
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace


# Spawns two simple tasks and specifies
# simple dependency between them.
def task_simple_dependency():
  task = TaskSpace("SimpleTask")
  @spawn(task[0]) 
  def t0():
    print("\tTask[0]")

  # Execute after task[0] completed.
  @spawn(task[1], [task[0]])
  def t1():
    print("\tTask[1]")
  # Return the last task.
  return task[1] 


# Spawns five series of tasks through the loop,
# and specifies all previous tasks as predecessors
# of each task.
def task_loop_dependency():
  NUM_SPAWNED_TASKS = 5
  task = TaskSpace("LoopTask")
  for i in range(NUM_SPAWNED_TASKS):
    # Spawns five tasks, and waits until all the
    # previous tasks are completed.
    @spawn(task[i], [task[:i]])
    def t():
      print("\tTask[",i,"]")
  # Return the last task.
  return task[NUM_SPAWNED_TASKS - 1]


# Spawns two simple tasks on differen task spaces
# and specifies simple dependency between them.
def different_taskspace_dependency():
  first_task = TaskSpace("FirstTask")
  @spawn(first_task)
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


def main():
  @spawn(placement=cpu)
  async def start_tasks():
    print("Task dependency: T[0] -> T[1] [START]")
    # Awaits the last task of the first example.
    await task_simple_dependency()
    print("Task dependency: T[0] -> T[1] [Done]\n")

    print("Task dependency: All previous tasks -> T[curr]: [START]")
    # Awaits the last task of the second example.
    await task_loop_dependency()
    print("Task dependency: All previous tasks -> T[curr]: [DONE]\n")

    print("Task dependency: T1[0] -> T2[1] [START]")
    # Awaits the last task of the third example.
    await different_taskspace_dependency()
    print("Task dependency: T1[0] -> T2[1] [DONE]")


if __name__ == "__main__":
  with Parla():
    main()
