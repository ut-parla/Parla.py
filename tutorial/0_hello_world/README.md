# Lesson 0: Hello, World!

In this lesson we create our first Parla program and print from a Parla task.
There is only one source code file for this lesson: `hello.py`
If Parla is installed properly, you can run this example as you would any other Python script:

```
python hello.py
```

Parla uses the task based parallel programming model, so we first introduce its main unit of work: tasks.
<!-- Everything you do in Parla is centered around tasks.  -->
In Parla, tasks are blocks of code, annotated by the programmer, to run asynchronously with respect to
their enclosing block.
Tasks may have dependencies, run in parallel with other tasks, run asynchronously, or run even on different hardware given various compute and memory constraints.
We'll give more detail on how to configure tasks for this in a later lesson.

For now, we define a single, simple task. We will break down this program line by line.

First, lines 1-3:

```
1  from parla import Parla
2  from parla.cpu import cpu
3  from parla.tasks import spawn
```

Line 1 imports the Parla runtime itself.
Line 2 imports and configures a type of device for the scheduler to dispatch
tasks to. Here we import 'cpu' to run on the Host machine.
Line 3 imports the `@spawn` decorator, which is the keyword for task creation in Parla.

Next, we look at lines 11-13:

```
11  if __name__ == "__main__":
12      with Parla():
13          main()
```

Line 12 invokes the Parla runtime to run our code. This creates a threadpool and initializes the scheduler that will dispatch tasks. All Parla code must be run within this context manager.
Notice that we do not directly create Parla tasks in the global scope - instead we define a `main` function and create our tasks there.
It's best practice to contain all Parla calls in a lower scope than the global scope - more on this in a later lesson.

Lastly, we look at `main` itself:

```
5  def main():
6      @spawn()
7      def hello_world():
8          print("Hello, World!")
```

In Line 6, we call our `@spawn` decorator, the Parla semantic for creating a task.
In Line 7, we define a task, which is a lot like defining a Python function (though Parla tasks are not truly Python functions - more on this later as well).
Inside our task we make our print call.

Congratulations! You've run your first Parla program!
