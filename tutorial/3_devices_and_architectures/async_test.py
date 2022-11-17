import numpy
from parla import Parla
from parla.array import copy, clone_here
from parla.cpu import cpu
from parla.tasks import spawn
from parla.task_collections import TaskSpace
from parla.function_decorators import specialized




def main():
    outer = TaskSpace("A")
    inner = TaskSpace("A")

    @spawn(placement=cpu)
    async def start_tasks():

        @spawn(outer[0], placement=cpu)
        async def task1():
            print("Task 1 Start", flush=True)

            @spawn(inner[0])
            async def task2():
                print("Inner Running", flush=True)

            print("Inner: ", inner, flush=True)

            await inner
            print("Task 1 End", flush=True)

        @spawn(outer[1], dependencies=[outer[0]])
        async def task3():
            print("Task 3", flush=True)

        print("Inner: ", inner, flush=True)
        print("Outer: ", outer, flush=True)

if __name__ == "__main__":
  with Parla():
    main()
