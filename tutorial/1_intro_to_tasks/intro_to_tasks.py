from parla import Parla
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace


def print_tasks():
  task = TaskSpace("Task")
  for i in range(5):
    @spawn(task[i], [task[:i]])
    def t():
      print("Task[",i,"]")
  return task[4]


def main():
  @spawn(placement=cpu)
  async def start_tasks():
    await print_tasks()
 

if __name__ == "__main__":
  with Parla():
    main()
