
#Import Parla
from parla import Parla, spawn, TaskSpace
# Initialize cpu device
from parla.cpu import cpu


def taskspace_example():

    #Declare a TaskSpace with a label 'Your First TaskSpace'
    #The label does not need to be unique and is only for convenience and introspection
    task_space = TaskSpace('Your First TaskSpace')

    #TaskSpaces are used to define a group of tasks.
    #They can be indexed into with any hashable value (we recommend integers)
    #The TaskSpace object and the index together form the taskid.
    #The taskid must be unique, launching two tasks with the same id will lead to errors.
    #Below we spawn a task with the taskid of task_space[0]

    @spawn(taskid=task_space[0])
    async def task_0():
        print('I am task 0', flush=True)

    #The taskid is the first argument to spawn, and can be used without a keyword.
    @spawn(task_space[1])
    async def task_1():
        print("I am task 1", flush=True)


    #The index can be of arbitrary dimension
    @spawn(task_space[1, 0, 0, 0, 1])
    async def task_2():
        print("I am task (1, 0, 0, 0, 1)", flush=True)

    #The index can be of arbitrary hashable type.
    #(In Parla 0.2 any strings should be of length 1, this will be fixed in an upcoming release)
    @spawn(taskid=task_space[(0, "a")])
    async def task_3():
        print("I am task (0, \"a\")", flush=True)

    #Note that [(0, "a")] is treated the same as [0, "a"]. Using both in the same namespace is a value error (as the taskid is not unique)
    @spawn(taskid=task_space[0, "b"])
    async def task_3():
        print("I am task (0, \"b\")", flush=True)


    print("TaskSpace: ", task_space, flush=True)
    print("------------", flush=True)


if __name__ == '__main__':
    with Parla():
        taskspace_example()




