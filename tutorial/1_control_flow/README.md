# Lesson 1: Control Flow

## How to manage Parla tasks

Parla supports flexible task management.
Here we introduce features to organize tasks and control their order of execution.

### TaskSpaces

Tasks are organized through TaskSpaces.
A **TaskSpace** object is an indexible collection of tasks.
These are used as a namespace for task-ids.

A TaskSpace can be indexed by any hashable value, such as an integer, string, or a tuple. Although in Parla 0.2, please keep any strings to length 1. This indexing can be of arbitrary dimension. For ease of use and interpretability, we recommend sticking to sets of integers.

The use of TaskSpaces can be seen in `0_taskspaces.py`:

```python
    task_space = TaskSpace("Your First TaskSpace")

    @spawn(taskid=task_space[0])
    async def task_0():
        print('I am task 0', flush=True)
```

The 'taskid' argument to `@spawn` is the name of the task.

**This name as a (TaskSpace + Index) combination must be unique.**
If a taskid is not specified, the task will be placed into the default TaskSpace.

`0_taskspaces.py` shows different ways to add a task to a TaskSpace.

### Task Dependencies

Tasks can be ordered by specifying dependencies between them.
If TaskB depends on TaskA then TaskB will be guaranteed to execute after TaskA has completed.

Dependencies can be specified by passing a list of taskids to the `dependencies` argument of `@spawn`. This is the second argument and may be specified without the keyword.

```python
    space_A = TaskSpace("TaskSpace A")
    @spawn(space_A[0])
    async def task_A():
        print('I am taskA', flush=True)

    @spawn(space_A[1], dependencies=[space_A[0]])
    async def task_B():
        print('I am taskB', flush=True)
```

Dependencies can be specified across any number of TaskSpaces.
Dependencies do not need to be within the same TaskSpace.

Dependencies can also be specified by slicing a TaskSpace.

```python

    @spawn(space_B[0], dependencies=[space_A[0:2]])
    async def task_B():
        print('I am taskB', flush=True)
```

The above task_B will depend on the first 2 entries of TaskSpace `space_A`.

Dependencies do not need to have been spawned at the time their ids are specified to a task that depends on them.

Examples of different ways to specify task dependencies can be seen in `1_dependencies.py`.

### Barriers

Barriers can also be used to synchronize execution of tasks. Barriers can only be used within tasks.

Parla uses Python's asyncio `async/await` semantics to wait on task execution. This means any task that contains a barrier must be declared `async def` and spawned normally with `@spawn`.

```python
    @spawn(taskid=task_space[0])
    async def task_0():
        print('I am task 0', flush=True)
        await tasks
        print('I am task 0 again', flush=True)
```

Although an `async def` task has an asynchronous body, the task will still wait for its body to fully complete before the task is finished.

Barriers are a special (implicit) type of dependency. When a task encounters a barrier, it will release control of its worker thread and spawn a new continuation of itself as a separate task.
This new task will have the targets of the barrier added to its dependency list. The continuation will only execute after all tasks listed in the barrier have completed.

The use of barriers can be seen in `2_barriers.py`.
One can wait on individual taskid, on a TaskSpace, or on a task handle.
