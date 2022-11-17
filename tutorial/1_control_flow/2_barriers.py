
#Import Parla
from parla import Parla, spawn, TaskSpace
# Initialize cpu device
from parla.cpu import cpu


def barrier_example():

    #Due to python async/await syntax, we need an outer asynchronous context to use the 'await' keyword.
    #This is done by spawning a main async task with the @spawn decorator.
    
    @spawn()
    async def main_task():

        space_A = TaskSpace('TS_A')
        space_B = TaskSpace('TS_B')

        #TaskSpaces, taskids, and Task handles are awaitable
        #This means that you can use them as barriers to wait for tasks to complete

        #Spawn tasks in space_A
        for i in range(4):
            @spawn(space_A[i])
            async def task_in_A():
                print(f'I am task {i} in TS_A', flush=True)
        
        #Wait for all tasks in space_A to complete
        #This ends the main_task and spawns a continuation task with dependencies on all tasks in space_A
        await space_A

        print('Everything in TS_A is guaranteed to be completed.', flush=True)

        #Spawn tasks in space_B
        for i in range(4):
            @spawn(space_B[i])
            async def task_in_B():
                print(f'I am task {i} in TS_B', flush=True)
        
        #Individual members of task spaces can be awaited.
        await space_B[0]
        print('Task 0 in TS_B is guaranteed to be completed.', flush=True)

        #The handle used to spawn the task can also be awaited.
        #As this definition corresponds to the last task launched, it is the same as awaiting space_B[3] here.
        await task_in_B
        print('Task 3 in TS_B guaranteed to be completed.', flush=True)

if __name__ == '__main__':
    with Parla():
        barrier_example()


    

