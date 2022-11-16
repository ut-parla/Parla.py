# Lesson 2: Scoping and Returns

This section introduces further detail on how to use tasks.
Specifically, we discuss how tasks are different than normal python functions, how one could return variables created within a task, and some general advice for writing performant Parla applications.

## Semantics of Variable Capture

As mentioned in Lesson 0, Parla tasks are NOT python functions.
Here we highlight that they modify the usual Pythonic scoping and variable-capture semantics.

When spawned Parla tasks capture variables from the enclosing scope by value.
If a variable is reassigned in the outer scope after spawning a task, it will not affect the value observed within the task.

This capture makes it possible to spawn tasks in a loop where each task holds values from the iteration that spawned it. This preserves sequential semantics when writing parallel code.

We've actually been using this all along in the tutorial, but to make it explicit consider the following chain of tasks:

```python
   for i in range(3):
       # Define a dependency chain
       dep = [T[i - 1]] if i > 0 else []

       #Spawn a chain of tasks
       @spawn(T[i], dependencies=dep)
       def task():
           # Reference a nonlocal variable 'i' (copied by value)
           nonlocal i
           # Add 4 to the now-local variable 'i'
           i = i + 4
           # Output to the console the value of the local 'i'
           print("Check Task State [", i, "]", flush=True)
```

Parla ensures that the output is `4 5 6` not `4, 8, 12`, `6, 7, 8`, or undefined behavior.

**The capture-by-value is a shallow copy**. For non-primitive types any underlying data buffers will be shared.

```python
   import numpy as np

   shared_array = np.zeros(3)
   for i in range(3):
       @spawn(T[i])
       def task():
           shared_array[i] = i

    await T
    print(shared_array, flush=True)
```

All tasks have a local shallow-copy of the `shared_array` NumPy ndarray.
The tasks all share the same data buffer and can update it concurrently. These changes are reflected in the outer-scope.

As tasks can write to a shared buffer, the user must be careful of race conditions when defining task dependencies!

Note: Parla can only capture local variables.
Any variables in the global namespace will be captured by reference and are vulnerable to race conditions and undefined behavior. For most use-cases, we recommend only defining Parla tasks in a local scope.

## Returning values from Tasks

The most common way to return data from a task is to write back to a shared buffer, like the NumPy example above. For passing back general objects, you can write back to lists or dictionaries from the outer scope.

```python
   shared_dictionary = {}

   for i in range(3):
       @spawn(T[i])
       def task():
           local_array = np.random.rand(3)
           shared_dictionary[i] = local_array

    await T
    print(shared_dictionary, flush=True)
```

Tasks can also return values through a synchronization point. Individual tasks (either through the handle or the task-id) can be awaited to get the return value of the task.

```python
    for i in range(3):
         @spawn(T[i])
         def task():
              local_array = np.random.rand(3)
              return local_array

    array1 = await T[0]
    array2 = await T[1]
    array3 = await T[2]
```

At the moment, there are no data futures in Parla.

## General Advice for Writing Tasks

Unlike many Python tasking systems, Parla tasks are run within a _thread_-based environment.
All tasks execute within the same process and, unfortunately, share the same Python interpreter (if run with CPython). All tasks need to acquire the Python Global Interpreter Lock (GIL) to execute any lines of native Python code. This means any pure Python will execute serially and not show parallel speedup.

**Tasks only achieve true parallelism when they call out to compiled libraries and external code that releases the GIL**, such as Numpy, Cupy, or jit-compiled Numba kernels. Parla is well-suited for parallelism in compute-heavy domains, but less-suited to workloads that need to execute many routines with native-Python-implemented libraries (like SymPy).

To write code that performs well in Parla, tasks should avoid holding and accessing the GIL as much as possible.
For a 50ms task, the GIL should be held for less than 5\% of the total task-time to avoid noticeable overheads.

Launching tasks with threads, however, does give us some advantages. Tasks share the same address space, allowing copyless operations on any memory buffers. We do not need to worry about managing or importing separate module lists in different persistent-Python processes, and any `jit` compilation by
Numba or other external libraries will be automatically reused between subsequent tasks.
