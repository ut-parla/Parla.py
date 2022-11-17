
#Import Parla
from parla import Parla, spawn, TaskSpace
# Initialize cpu device
from parla.cpu import cpu


def dependency_example_1():

    space_A = TaskSpace('TS_A')

    #Spawn tasks in space_A
    @spawn(space_A[0])
    async def task_0_in_A():
        print(f'I am task {0} in TS_A', flush=True)
    
    #Task space_A[1] depends on task_0_in_A.
    #This means that space_A[0] must complete before space_A[1] can begin.
    @spawn(space_A[1], dependencies=[space_A[0]])
    async def task_1_in_A():
        print(f'I am task {1} in TS_A', flush=True)

def dependency_example_2():

    space_A = TaskSpace('TS_A')
    space_B = TaskSpace('TS_B')

    #Spawn tasks in space_A
    for i in range(4):
        @spawn(space_A[i])
        async def task_0_in_A():
            print(f'I am task {i} in TS_A', flush=True)
    
    #Spawn task in space_B with dependencies in space_A
    # - Dependencies can be specified as a list of taskids across any number of task spaces
    # - Dependencies can be specified as a slice of a task space
    @spawn(space_B[0], dependencies=[space_A[0:2]])
    async def task_0_in_B():
        print(f'I am task {0} in TS_B', flush=True)



if __name__ == '__main__':
    print("Running dependency example 1:", flush=True)
    with Parla():
        dependency_example_1()
    
    print("Running dependency example 2:", flush=True)
    with Parla():
        dependency_example_2()


    

