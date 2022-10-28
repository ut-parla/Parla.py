'''
Parla: more_on_tasks.py

Sample Output:

Before Task[ 0 ]
Before Task[ 1 ]
Before Task[ 3 ]
Before Task[ 2 ]
Before Task[ 4 ]
---------
After Task[ 5 ]
After Task[ 6 ]
After Task[ 9 ]
After Task[ 7 ]
After Task[ 8 ]
---------
Check Task[ 4 ]
Check Task[ 5 ]
Check Task[ 6 ]

This is a Parla tutorial that emphasizes control flow and scoping semantics when dealing with multiple tasks. In terms of control flow, tasks can be 'awaited' (or blocked) until other groups of tasks complete.  In terms of scoping semantics, variables are copied by value, but data structures are copied by reference.

Awaiting tasks are covered in lines 39-74 and a scoping-semantics example is covered in lines 82-101.

'''

# Import Parla context
from parla import Parla

# Import for placing tasks on the cpu
from parla.cpu import cpu

# Import for the spawn decorator and TaskSpace declaration
from parla.tasks import spawn, TaskSpace

# Define the main function (required of all Parla implementations)
def main():

    # Spawn a task, 'placing' it on the cpu
    # Note the 'async' keyword
    @spawn(placement=cpu)
    async def task_launcher():

        # Declare the TaskSpace and call it 'TaskSpace1'
        taskSpace = TaskSpace("TaskSpace1")

        # Loop for launching 5 tasks
        for i in range(5):

            # Spawn a subtask, giving it an id index from the loop
            @spawn(taskid=taskSpace[i])
            def task():

                # Print a statement to console, flushing-out the print buffer
                print("Before Task[", i, "]", flush=True)

        # Wait for all tasks spawned in 'taskSpace' to complete
        await taskSpace

        # Print a console output divider
        print('---------')

        # Loop for spawning some more tasks
        for i in range(5, 10):

            # Spawn a subtask, giving it an id index from the loop
            @spawn(taskid=taskSpace[i])
            def task():

                # Print a statement to console, flushing-out the print buffer
                print("After Task[", i, "]", flush=True)

        # Wait for all tasks spawned in 'taskSpace' to complete
        await taskSpace

        # Print a console output divider
        print('---------')

        # Declare another TaskSpace and call it 'TaskSpace2'
        taskSpace2 = TaskSpace("TaskSpace2")

        # Loop for launching 3 tasks
        for i in range(3):

            # Define a dependency list for the upcoming task with one item
            # (the previously-spawned task)
            dep = [taskSpace2[i - 1]] if i > 0 else []

            # Spawn a subtask, giving it an id index from the loop and a
            # dependency list of 'dep'
            @spawn(taskid=taskSpace2[i], dependencies=dep)
            def task():

                # Reference a nonlocal variable 'i' (copied by value)
                nonlocal i

                # Add 4 to the now-local variable 'i'
                i = i + 4

                # Output to the console the value of the local 'i'
                print("Check Task[", i, "]", flush=True)


# Execute the Parla program with the Parla Context
if __name__ == "__main__":
    with Parla():
        main()
