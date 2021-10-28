# Lesson 0: Hello, World!

In this lesson, we will create our first Parla program and print to the console from a Parla task. There is only one source code file for this lesson: `hello.py`, and if Parla is installed properly, you can run this example as you would any other Python script:

```
$ python hello.py
```

Parla uses the task-based parallel programming model, and here we first introduce its main unit of work: tasks. Everything you do in Parla is centered around tasks--"tasks" being defined as blocks of code that are annotated by the programmer to run asynchronously with respect to their enclosing block.  Tasks may have dependencies, may run asynchronously in parallel with other tasks, or may even run on different hardware given various computation and memory constraints.  We will discuss more details on how to configure tasks for these various approaches in a later lesson. For now, we define a single, simple task. We will break down this program line by line.

First, lines 2 and 5:

```
2  from parla import Parla
...
5  from parla.tasks import spawn
```

Line 2 imports the Parla runtime itself, and line 5 imports the `@spawn` decorator, which is the main mechanism for instantiating tasks in Parla.

We'll temporarily skip ahead to lines 18-20:

```
18  if __name__ == "__main__":
19      with Parla():
20          main()
```

In line 18, we instruct the Python runtime to execute this block of code only if running as the main file from the command line (and not from something else such as an import statement in another Python file).

Line 19 invokes the Parla runtime to run our code--it creates a threadpool and initializes the scheduler that will dispatch tasks. All Parla code must be run within this context manager. Notice that we do not directly create Parla tasks in the global scope. Instead, we define a `main` function and create our tasks there, and it is best practice to contain all Parla calls in a lower scope than the global scope (more on this in a later lesson).

And finally, line 20 calls our `main` function defined on lines 8-15:

```
8  def main():
9  
10     # ...
11     @spawn()
12     def hello_world():
13 
14         # ...
15         print("Hello, World!")
```

On line 11, we call our `@spawn` decorator, the Parla semantic for creating a task, and we decorate the function on line 12 "hello_world()" with this decorator. Keep in mind, however, that Parla tasks are not truly Python "functions" - more on this later as well.  Inside our task, we simply print the statement "Hello, World!" to the console.

As stated previously, you can run this Python file as you would any other Python script with `$ python hello.py`.  Once run, you should see "Hello, World!" the console output.

Congratulations! You've run your first Parla program!
