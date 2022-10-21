# Lesson 0: Hello, World!

We create our first Parla program and print to the console from a Parla task. 

There is only one source code file for this lesson: `hello.py` 

If Parla is installed properly, you can run this example as you would any other Python script:

```bash
$ python hello.py
```

Parla's parallel programming model is centered around ***tasks***.
Tasks are a basic unit of work. Specifically in Parla, tasks are annotated code blocks that run asynchronously with respect to their enclosing block.


Although syntactically tasks often look like Python functions, they are semantically different.
Parla tasks are not functions in the usual sense: the Parla runtime can launch tasks as soon as they are spawned; they capture local variables in the closure by value; they cannot be called and, typically, do not return values.


Tasks: 
- can be defined with dependencies. This creates an online & dynamic workflow DAG that begins execution as it is spawned.
- may run asynchronously and concurrently with other tasks.
- may have various constraints (device, memory, data, threads) on where and when they can execute.
- can launch on different hardware contexts.
- can specify input and output data objects to prefetch required data into device memory. 


 We will discuss more advantaged usage and features of tasks later. For now, we define a single, simple task.

We break down this program line by line:

First, lines 13 - 19:

```python
13  from parla import Parla, spawn
...
15  from parla.cpu import cpu
```

Line 13 imports the context manager for the Parla runtime and the `@spawn` decorator for instantiating tasks. 

Line 15 loads and configures the `cpu` device type which is needed to have a valid target for task execution. 

We'll temporarily skip ahead to lines 28 - 30:

```python
28  if __name__ == "__main__":
29      with Parla():
30          main()
```

Line 29 enters and starts the Parla runtime.
This creates a threadpool and begins running a scheduler that will listen for spawned tasks.
The scheduler can dispatch tasks to all configured device types (defined at import time) before it is initialized. 

**All Parla code must execute within the Parla context manager.**

Notice that we do not directly create Parla tasks in the global scope. Instead, we define a `main` function and create our tasks there. *To ensure correct capture of local variables, Parla tasks should not be defined in the global scope.*

And finally, our `main` function defined on lines 20 - 25:

```python
20 def main():
21 
22     # ...
23     @spawn()
24     def hello_world():
25        print("Hello, World!", flush=True)
```

On line 25, the `@spawn` decorator creates a task that executes the function it decorates. 
As soon as the task is spawned, it is visible to the runtime and may begin asynchronous execution. 

In this example, the task is not guaranteed to complete until the Parla context is exited on line 31 (which waits for all unfinished tasks). 

Once run, you should see "Hello, World!" the console output.
Congratulations! You've run your first Parla program!
