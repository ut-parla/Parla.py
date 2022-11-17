'''
This is a Parla tutorial that emphasizes scoping semantics and returning values.
'''

# Import Parla context
from parla import Parla

# Import for placing tasks on the cpu
from parla.cpu import cpu

# Import for the spawn decorator and TaskSpace declaration
from parla.tasks import spawn, TaskSpace

import numpy as np


def local_capture():

    @spawn()
    async def main_task():

        print("Local Capture Semantics", flush=True)
        print("--------", flush=True)
        T = TaskSpace("TaskSpace A")
        for i in range(3):
            # Define a dependency chain
            dep = [T[i - 1]] if i > 0 else []
            #Spawn a chain of tasks
            @spawn(T[i], dependencies=dep)
            async def task(): 
                # Reference a nonlocal variable 'i' (copied by value)
                nonlocal i
                # Add 4 to the now-local variable 'i'
                i = i + 4
                # Output to the console the value of the local 'i'
                print("Check Task State [", i, "]", flush=True)

        await T
        print()

        print("Captures are a Shallow-Copy:", flush=True)
        print("--------", flush=True)

        shared_array = np.zeros(3)
        T = TaskSpace("TaskSpace B")
        for i in range(3):
            @spawn(T[i])
            async def task(): 
                shared_array[i] = i

        await T
        print("Shared Array:", shared_array, flush=True)
        print()


def local_return():

    @spawn()
    async def main_task():

        print("Tasks can pass objects to the outer scope", flush=True)
        print("--------", flush=True)

        shared_dictionary = {}
        T = TaskSpace("TaskSpace C")
        for i in range(3):
            # Define a dependency chain
            dep = [T[i - 1]] if i > 0 else []
            #Spawn a chain of tasks
            @spawn(T[i], dependencies=dep)
            async def task(): 
                local_array = np.zeros(3)+i
                shared_dictionary[i] = local_array

        await T
        print("Shared Dictionary:", shared_dictionary, flush=True)
        print()

        print("Tasks can return values through await:", flush=True)
        print("--------", flush=True)

        T = TaskSpace("TaskSpace D")
        for i in range(3):

            @spawn(T[i])
            async def task(): 
                local_array = np.zeros(3)+i
                return local_array

        array1 = await T[0]
        array2 = await T[1]
        array3 = await T[2]

        print("Array 1:", array1, flush=True)
        print("Array 2:", array2, flush=True)
        print("Array 3:", array3, flush=True)
        print()


# Execute the Parla program with the Parla Context
if __name__ == "__main__":
    with Parla():
        local_capture()

    with Parla():
        local_return()
