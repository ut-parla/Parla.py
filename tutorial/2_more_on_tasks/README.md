# Tutorial 2: More on Tasks

This tutorial is meant to introduce further detail on the specifics of using Parla tasks.
More specifically, it demonstrates how to control the execution flow of your program,
how tasks are different than normal python functions, and presents some general advice for running tasks.

## Example 1: Control Flow
Parla uses the Python asynchronous `async`/`await` syntax to control and block program flow,
and Parla task objects, task ids, task collections, and `TaskSpaces` are all awaitable.
Let's look at the first half of the main function at lines 37 - 72:

```
37 # Define the main function (required of all Parla implementations)
38 def main():
39 
40     # Spawn a task, 'placing' it on the cpu
41     # Note the 'async' keyword
42     @spawn(placement=cpu)
43     async def task_launcher():
44 
45         # Declare the TaskSpace and call it 'TaskSpace1'
46         taskSpace = TaskSpace("TaskSpace1")
47 
48         # Loop for launching 5 tasks
49         for i in range(5):
50 
51             # Spawn a subtask, giving it an id index from the loop
52             @spawn(taskid=taskSpace[i])
53             def task():
54 
55                 # Print a statement to console, flushing-out the print buffer
56                 print("Before Task[", i, "]", flush=True)
57 
58         # Wait for all tasks spawned in 'taskSpace' to complete
59         await taskSpace
60 
61         # Print a console output divider
62         print('---------')
63 
64         # Loop for spawning some more tasks
65         for i in range(5, 10):
66 
67             # Spawn a subtask, giving it an id index from the loop
68             @spawn(taskid=taskSpace[i])
69             def task():
70 
71                 # Print a statement to console, flushing-out the print buffer
72                 print("After Task[", i, "]", flush=True)
```

The line to take note of in this segment of code is specifically line 59, where execution of the `task_launcher()` function
is blocked until all tasks defined in the task space, `TaskSpace1` (taskSpace[0] - taskSpace[4]) complete.
Each of these tasks are responsible for printing to the console "Before Tasks" with their ID
(flush is used here simply as a precautionary measure so that all output is appropriately displayed
in the console before proceeding). After these tasks from `TaskSpace1` are completed,
the code continues and spawns all subsequent tasks. These other tasks are responsible for printing to the
console "After Tasks" with their ID. It can be noted in the program's output that all "Before Tasks" are
printed before any of the "After Tasks".


## Example 2: Scoping Semantics
As mentioned in Tutorial 0, Parla tasks are NOT python functions--they are defined in a similar fashion and can take arguments,
but in doing-so (that is, in defining them) they modify the usual scoping and variable-capture semantics.

Parla tasks capture variables from the enclosing scope by value when spawned.
If a variable is reassigned in the outer scope after spawning a task,
it will not affect the variable value observed within the task. This makes it possible to
spawn tasks in loops and have each task use the values of local variables from the iteration
that spawned it, thus maintaining sequential semantics!
We've actually been using this all along in Example 1 of this tutorial.
Conversely, data structures referenced by variables are not copied--this means that arrays are
still shared between tasks.

These semantics allow code to be written similar to the following (on lines 83 - 102):

```
83         # Loop for launching 3 tasks
84         for i in range(3):
85 
86             # Define a dependency list for the upcoming task with one item
87             # (the previously-spawned task)
88             dep = [taskSpace2[i - 1]] if i > 0 else []
89
90             # Spawn a subtask, giving it an id index from the loop and a
91             # dependency list of 'dep'
92             @spawn(taskid=taskSpace2[i], dependencies=dep)
93             def task():
94 
95                 # Reference a nonlocal variable 'i' (copied by value)
96                 nonlocal i
97 
98                 # Add 4 to the now-local variable 'i'
99                 i = i + 4
100
101                # Output to the console the value of the local 'i'
102                print("Check Task[", i, "]", flush=True)
```

In this way, we get `4 5 6` for the "Check Task" output instead of `4, 8, 12`, `6, 7, 8`, or other
likely-undefined behavior based on when the tasks are launched while the loop is running.

Note: In the current version of Parla this capture by value only happens to locally defined variables.
Any variables in the global namespace will be captured by reference and are vulnerable to the undefined
behavior mentioned above. For most use-cases, we recommend keeping Parla tasks away from the global namespace.

## Advice for Writing Tasks
Unlike many Python tasking systems, Parla tasks are defined in a multithreading environment.
This means tasks execute within the same process and, perhaps unforunately, share the same Python interpreter.

This means that all tasks need to acquire the Python Global Interpreter Lock (GIL) to execute native
Python code, and this implies that the code is executed serially. Tasks only achieve true parallelism
when they call out to libraries and external code that releases the GIL, and this includes libraries
such as Numpy or Cupy. This makes it well-suited for parallelism in compute-heavy
domains, but less-suited to workloads that need to execute disparate sections of native-Python-implemented
libraries. To write code that performs well in Parla, tasks should be written to access--and More
importantly, hold--the GIL as little as possible.

Launching tasks with threads, however, does give us some advantages. The tasks share the same
address space, allowing copyless operations on memory buffers like arrays. We do not need to worry
about seperate module lists in different persistent-Python processes, and any `jit` compilation by
Numba or other external libraries will be automatically reused over subsequent tasks.
