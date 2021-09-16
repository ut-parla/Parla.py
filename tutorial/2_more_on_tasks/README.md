#Tutorial 2: More on Tasks

This section of the tutorial is to give a bit more detail on the particulars of
using tasks. Specifically: how to control the execution flow of your program,
how tasks are different than normal python functions, and some general advice
for running tasks.

## Control Flow
Parla uses the asynchronous `async`/`await` syntax to control and block program
flow. Parla task objects, task ids, task collections, and `TaskSpaces` are all
awaitable.

```
def main():
  task = TaskSpace("Task")
  for i in range(5):
    @spawn(task[i])
    def t2():
      print("Before Task[",i,"]", flush=True)

  await task

  for i in range(5, 10):
    @spawn(task[i])
    def t2():
        print("After Task[", i, "]", flush=True)

```

In the above example the code will await on all tasks currently in the
TaskSpace before execution of the main thread continues. All "Before" tasks will finish before "After" tasks.


## Scoping Semantics
As mentioned in the first tutorial, Parla tasks are NOT python functions. They
are defined similarly and can take arguments, but they they modify the usual scoping and variable capture semantics.

Parla tasks capture variables from the enclosing scope by value when spawned. If
a variable is reassigned in the outer scope after spawning a task, it will not
affect the variable value observed within the task.

This makes it possible to spawn tasks in loops and have each task use the
values of local variables from the iteration that spawned it, thus maintaining sequential semantics!
We've actually been using this all along in the past example.

Data structures referenced by variables are not copied, so arrays are still
shared between tasks.

This allows us to write code like :

```
for i in range(3):
    @spawn(task[i], task[i-1])
    def worker():
        i = i + 4
        print(i, flush=True)
```

and get `4 5 6` instead of `4, 8, 12`, `6, 7, 8`, or other likely undefined
behavior based on when the tasks launch while the loop is running.

Note: In the current version of Parla this capture by value only happens to
locally defined variables. Any variables in the global namespace will be
captured by reference and are vulnerable to the undefined behavior mentioned
above. For most usecases, we recommend keeping Parla tasks away from the global
namespace.


## Advice for Writing Tasks
Unlike many Python tasking systems, Parla tasks are defined in a multithreading
environment. This means tasks execute within the same process and, perhaps
unforunately, share the same Python interpreter.

This means that all tasks need to acquire the Python Global Interpreter Lock
(GIL) to execute native Python code and that it is executed serially.
Tasks only achieve true parallelism when they call out to libraries and
external code that releases the GIL, such as Numpy, Cupy, .
This makes it well structured for parallelism in domains with compute heavy
domains, but less suited to workloads that need to execute disparate sections
of native Python implemented libraries.

To write performant code in Parla tasks should be written to access, and more
importantly hold, the GIL as little as possible.

Launching tasks with threads does give us some advantages.
The tasks share the same address space allowing copyless operations on memory
buffers like arrays, we do not need to worry about seperate module lists in
different persistent Python processes,  and any `jit` compilation by Numba or external libraries
will be automatically reused over subsequent tasks.



