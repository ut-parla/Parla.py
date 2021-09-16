# Lesson 1: Introduction to Parla Tasks

Parla supports flexible task management features. This lesson introduces
task space which is a collection of tasks with IDs, and the way to specify
dependencies among tasks, through a simple Parla example: `intro_to_tasks.py`

You can run this example by the below command:

```
python intro_to_tasks.py
```

This script spawns five tasks, assigns task ids from task[0] to task[4],
and each of them prints its task ID. In this case, a task waits on tasks
having small indices through Parla task dependency decorator.

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
6  task = TaskSpace("Task")
```

Line 6 declares a `TaskSpace` named 'task'.
At this point, no task is placed in and no slicing happens on the `TaskSpace`.

Next, we look at lines 7-10:

```
 7  for i in range(5):
 8    @spawn(task[i], dependencies=[task[0:i-1]])
 9    def t():
10      print("Task[",i,"]")
```

Line 7-8 spawns five tasks with `@spawn` decorator and its two parameters.
In this case, the first parameter, `task[i]`, assigns an index `i` to each task.
This loop finally slices the `TaskSpace` into 5 dimensional task spaces.

The second parameter, `[task[0:i-1]]`, specifies dependencies among tasks.
This can be a list of any combination of tasks and collections of tasks.
Here by slicing the current `TaskSpace` up to the current tasks, we produce a series of tasks where each depends on all previous tasks.
<!--- Note that this is equivalent to just depending on the previous task --->
Each task will be scheduled and be run after all the previous tasks are completed, and
therefore, the output will print task ids in increasing order.

Congratulations! You've learned `TaskSpace` and specification for task dependencies of Parla.
