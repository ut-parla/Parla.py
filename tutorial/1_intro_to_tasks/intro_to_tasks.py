from parla import Parla
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace

def main():
  task = TaskSpace("Task")
  for i in range(5):
    @spawn(task[i], [task[0:i-1]])
    def t():
      print("Task[",i,"]")

if __name__ == "__main__":
  with Parla():
    main()
