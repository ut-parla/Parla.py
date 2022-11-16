# Lesson 1: Control Flow

## How to manage Parla tasks

Parla supports flexible task management.
Here we introduce features to organize tasks and control their order of execution.

### TaskSpaces

Tasks are organized through TaskSpaces, an indexible collection of tasks.
You can think of these as a kind of namespace for task ids.

A TaskSpace can be indexed by any hashable value. Although, for ease of use and interpretability, we highly recommend sticking to sets of integers. TaskSpaces can be indexed in arbitrary dimension.

In Parla 0.2, please only use strings of length 1 for TaskSpace keys.

The use of TaskSpaces can be seen in `0_taskspaces.py`:

```python
    task_space = TaskSpace("Your First TaskSpace")

    @spawn(taskid=task_space[0])
    async def task_0():
        print('I am task 0', flush=True)
```

The 'taskid' argument to `@spawn` is the name of the task. **This name as a (TaskSpace + Index) combination must be unique during the lifetime of the Parla runtime.**
If a taskid is not specified, the task will be placed into a default global TaskSpace.

`0_taskspaces.py` shows different ways to add a task to a TaskSpace.

### Task Dependencies

Tasks can be ordered by specifying dependencies between them.
If TaskB depends on TaskA then TaskB will be guaranteed to execute only after TaskA has completed.

Dependencies can be specified by passing a list of taskids to the `dependencies` argument of `@spawn`. This is the second argument and may be specified without a keyword.

```python
    space_A = TaskSpace("TaskSpace A")
    @spawn(space_A[0])
    async def task_A():
        print('I am taskA', flush=True)

    @spawn(space_A[1], dependencies=[space_A[0]])
    async def task_B():
        print('I am taskB', flush=True)
```

Dependencies can be specified across any number of TaskSpaces. Tasks do not need to be within the same TaskSpace to depend on each other.

TaskSpaces can be sliced with standard Python syntax.

```python

    @spawn(space_B[0], dependencies=[space_A[0:2]])
    async def task_C():
        print('I am taskC', flush=True)
```

For example, the above task_C will depend on the first 2 entries of TaskSpace `space_A`.

Tasks can be spawned out of order. A tasks dependencies do not need to have been before being listed through the TaskSpace.

Examples of different ways to specify task dependencies can be seen in `1_dependencies.py`.

### Barriers

Barriers are used to synchronize execution of tasks. They block a task's execution until all tasks listed in the barrier have completed.

Parla uses Python's asyncio `async/await` semantics to wait on task execution. This means any task that contains a barrier must be declared `async def` and spawned normally with `@spawn`.

```python
    @spawn(taskid=task_space[0])
    async def task_0():
        print('I am task 0', flush=True)
        await tasks
        print('I am task 0 again', flush=True)
```

Although it looks like an `async def` task has an asynchronous body, the task will still wait for its body to fully complete before the task is marked as completed. Tasks can wait on tasks that are not marked `async def`.

Barriers are a special (implicit) type of dependency. When a task encounters a barrier, it will release control of its worker thread and spawn a new continuation of itself as a separate task.
As a consequence, Barriers can only be used within tasks and cannot be used in the outermost non-task scope.

This new task will have the targets of the barrier added to its dependency list. The continuation will only execute after all tasks listed in the barrier have completed.

The use of barriers can be seen in `2_barriers.py`.
One can wait on individual taskid, on a TaskSpace, or on a task handle.
