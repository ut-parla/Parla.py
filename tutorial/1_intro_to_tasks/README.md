# Lesson 1: Introduction to Parla Tasks

Parla supports flexible task management features. This lesson introduces
task space which is a collection of tasks with IDs, the way to specify
dependencies among tasks, and a feature of awaiting a task through a
simple Parla example: `intro_to_tasks.py`

You can run this example by the below command:

```
python intro_to_tasks.py
```

This script spawns five tasks, assigns task ids from task[0] to task[4],
and each of them prints its task ID. In this case, a task waits on tasks
having small indices through Parla task dependency decorator.
This script waits for completion of the last priority task.

The below is outputs of the example.


```
Task[ 0 ]
Task[ 1 ]
Task[ 2 ]
Task[ 3 ]
Task[ 4 ]
```

Let's break down and understand this script line by line.
In this lesson, we will skip lines explained by previous lessons.

First, line 3:

```
3  from parla.tasks import spawn, TaskSpace
```

Line 3 imports `@spawn` decorator and `TaskSpace` class.
A `TaskSpace` provides an abstract high dimensional indexing space in which tasks can be placed.
A `TaskSpace` can be indexed using any hashable values and any number of indices.
If a dimension is indexed with numbers then those dimensions can be sliced.

Further lines show the actual usage of the `TaskSpace`.

```
7  task = TaskSpace("Task")
```

Line 7 declares a `TaskSpace` named 'Task'.
At this point, no task is placed in and no slicing happens on the `TaskSpace`.

Next, we look at lines 8-11:

```
 8  for i in range(5):
 9    @spawn(task[i], dependencies=[task[:i]])
10    def t():
11      print("Task[",i,"]")
```

Line 8-9 spawns five tasks with `@spawn` decorator and its two parameters.
In this case, the first parameter, `task[i]`, assigns an index `i` to each task.
This loop finally slices the `TaskSpace` into 5 dimensional task spaces.

The second parameter, `[task[:i]]`, specifies dependencies among tasks.
This can be a list of any combination of tasks and collections of tasks.
Here by slicing the current `TaskSpace` up to the current tasks, we produce
a series of tasks where each depends on all previous tasks.
Each task will be scheduled and be run after all the previous tasks are completed, and
therefore, the output will print task ids in increasing order.

Finally, this script exploits `await` syntax of Python to wait the last task.

```
16  @spawn(placement=cpu)
17  async def start_tasks():
18    await print_tasks()
```

Lines 16-18 spawns a simple entrance task as an `async` function.
Line 18 awaits completion of `print_tasks()`.

```
12  return task[4]
```

In this case, `print_tasks()` returns the last index of the `TaskSpace` in line 12. 
Therefore, `start_tasks()` awaits the last task of `print_tasks()`.

In Parla, all tasks, task spaces, and task sets are awaitable.
This feature is useful when you want to block until the awaiting task have completed. 

Congratulations! You've learned `TaskSpace` and specification for task dependencies of Parla.
