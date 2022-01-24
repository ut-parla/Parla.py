'''

Parla: 1_intro_to_tasks.py

Sample Output:
    Task - Start
            task_0
    Task - End
    
    Task Dependency - Start
            task_0
            task_1
    Task Dependency - End
    
    Task Spawning Tasks In A Loop - Start
            general_task_0
            general_task_1
            general_task_2
            general_task_3
            general_task_4
    Task Spawning Tasks In A Loop - End
    
    Task Spawning Tasks In A Loop Without Specifying Dependencies - Start
            general_independent_task_0 with 9999900 iterations
            general_independent_task_1 with 9999000 iterations
            general_independent_task_3 with 9900000 iterations
            general_independent_task_2 with 9990000 iterations
            general_independent_task_4 with 9000000 iterations
    Task Spawning Tasks In A Loop Without Specifying Dependencies - End
    
    Task taskSpace dependency - Start
            Task from first taskSpace
            Task from second taskSpace
    Task taskSpace dependency - End

This is a Parla tutorial emphasizing the various ways in which one could
go about scheduling Parla tasks with dependencies. Note in example 3a
("Task Spawning Tasks In A Loop Without Specifying Dependencies" in the
example output above) how general_independent_task_3 completes before
general_independent_task_2.

Example 1  - Spawns a task in the simplest of ways

Example 2  - Spawns a task, but only after the specified dependency completes

Example 3  - Spawn tasks in a loop, where each tasks needs to wait for
             previously-spawned tasks to complete before beginning.

Example 3a - Spawn tasks as in example 3, but without specifying
             dependencies

Example 4  - Spawn tasks in another TaskSpace, but only after the tasks
             in a different TaskSpace have completed.

'''

# Import Parla
from parla import Parla

# Import for placing tasks on the cpu
from parla.cpu import cpu

# Import for the spawn decorator and TaskSpace declaration
from parla.tasks import spawn, TaskSpace


# Example 1 - Simple Task
def task_simple():
    
    # Declare the TaskSpace and call it 'SimpleTaskSpace'
    taskSpace = TaskSpace('SimpleTaskSpace')

    # Spawn a simple task and assign it a taskid of 0 within taskSpace
    @spawn(taskid=taskSpace[0])
    def task_0():
        print('\ttask_0')
    
    # Return the awaitable
    return taskSpace[0]


# Example 2 - Simple Task Dependency
def task_simple_dependency():

    # Declare a TaskSpace and call it 'SimpleTaskSpace2'
    taskSpace_2 = TaskSpace('SimpleTaskSpace2')

    # Spawn the first task
    @spawn(taskid=taskSpace_2[0])
    def task_0():
        print('\ttask_0')

    # Spawn the second task, but only after taskSpace_2[0] has completed
    @spawn(taskid=taskSpace_2[1], dependencies=[taskSpace_2[0]])
    def task_1():
        print('\ttask_1')

    # Return the awaitable
    return taskSpace_2[1]


# Example 3 - Spawning tasks in a loop
def task_spawn_tasks_in_a_loop():

    # Constant for number of tasks to spawn
    NUMBER_OF_TASKS_TO_SPAWN = 5

    # Declare a task space and call it 'LoopTaskSpace'
    taskSpace_3 = TaskSpace('LoopTaskSpace')

    # Loop to instantiate tasks
    for i in range(NUMBER_OF_TASKS_TO_SPAWN):

        # Spawn a task assigning it a taskid within taskSpace_3, but
        # only after the previous dependencies have completed
        @spawn(taskid=taskSpace_3[i], dependencies=[taskSpace_3[:i]])
        def general_task():
            print('\tgeneral_task_',i, sep='')

    # Return the last task for the call to "await"
    return taskSpace_3[NUMBER_OF_TASKS_TO_SPAWN - 1]


# Example 3a - Spawning tasks in a loop without specifying the dependencies
def task_spawn_tasks_in_a_loop_without_dependencies():

    # Constant for number of tasks to spawn
    NUMBER_OF_TASKS_TO_SPAWN = 5

    # Declare a task space and call it 'LoopTaskSpace'
    taskSpace_4 = TaskSpace('LoopTaskSpace2')

    # Loop to instantiate tasks
    for i in range(NUMBER_OF_TASKS_TO_SPAWN):

        # Spawn a task assigning it a taskid within taskSpace_4
        # NOTE: Contrasting with example 3, each task is spawned
        # without specifying dependencies
        @spawn(taskid=taskSpace_4[i])
        def general_independent_task():

            # Give the 'later' tasks a smaller problem
            # size to simulate 'future' tasks completing
            # before previously-spawned tasks
            iterations = 10000000 - pow(10, i + 2)

            # Output statement
            print('\tgeneral_independent_task_', i, ' with ', iterations, ' iterations', sep='')

            # Simulate busy work
            for j in range(iterations):
                pass

    # Return the last task for the call to "await"
    return taskSpace_4[NUMBER_OF_TASKS_TO_SPAWN - 1]


# Example 4 - Spawning tasks only after a previous taskSpace's tasks have completed
def task_different_taskSpace_dependency():

    # Declare a task space and call it 'FirstTaskSpace'
    first_taskSpace = TaskSpace('FirstTaskSpace')

    # Spawn a task in the first task space
    @spawn(first_taskSpace)
    def task_from_first_taskSpace():
        print('\tTask from first taskSpace')

    # Declare a second task space and call it 'SecondTaskSpace'
    second_taskSpace = TaskSpace('SecondTaskSpace')

    # Spawn a task in the second task space, but only after
    # the tasks in the first taskSpace have completed
    @spawn(second_taskSpace, [first_taskSpace])
    def task_from_second_taskSpace():
        print('\tTask from second taskSpace')
    
    # Return the awaitable second_taskSpace
    return second_taskSpace


# Define the main function (required of all Parla implementations)
def main():

    # Spawn a task, 'placing' it on the cpu
    @spawn(placement=cpu)
    async def start_tasks():

        # Test Example 1 
        print('Task - Start')
        await task_simple() # Wait for the task to complete
        print('Task - End\n')

        # Test Example 2
        print('Task Dependency - Start')
        await task_simple_dependency() # Wait for the task to complete
        print('Task Dependency - End\n')

        # Test Example 3
        print('Task Spawning Tasks In A Loop - Start')
        await task_spawn_tasks_in_a_loop() # Wait for the task to complete
        print('Task Spawning Tasks In A Loop - End\n')

        # Test Example 3a
        print('Task Spawning Tasks In A Loop Without Specifying Dependencies - Start')
        await task_spawn_tasks_in_a_loop_without_dependencies() # Wait for the task to complete
        print('Task Spawning Tasks In A Loop Without Specifying Dependencies - End\n')

        # Test Example 4
        print('Task taskSpace dependency - Start')
        await task_different_taskSpace_dependency() # Wait for the task to complete
        print('Task taskSpace dependency - End\n')


# Execute the Parla program with the Parla Context
if __name__ == '__main__':
    with Parla():
        main()
