# Lesson 1: Introduction to Parla Tasks

Parla supports flexible task management features. This lesson introduces the "TaskSpace" construct--an instantiable object that houses a collection of tasks with unique identifiers--that is, every task has a "taskid".  Parla uses this construct as a way of specifying dependencies among tasks, and is the main mechanism for "awaiting" for other tasks to complete before a task begins its own execution.

You can run this example by the below command:

```
python 1_intro_to_tasks.py
```

This script introduces five examples of the task dependency use cases. The first example simply spawns a task, and serves as a grounding for building the tutorial.

The second example spawns two tasks, assigns them taskid's of `taskSpace_2[0]` and `taskSpace_2[1]`, respectively. Each of these two tasks prints a statement denoting which task is executing, and the second task specifies that `taskSpace_2[1]` depends on `taskSpace_2[0]`.  In other words, `taskSpace_2[0]` needs to complete before `taskSpace_2[1]` can begin its execution, and this is specified through Parla's "@spawn" decorator, where the "dependencies" keyword argument lists the dependencies that `taskSpace_2[1]` needs have completed before it can begin. Therefore, `taskSpace_2[1]` will not run until `taskSpace_2[0]` completes.

Shown below is an example of Example 2's output:

```
Task Dependency - Start
        task_0
        task_1
Task Dependency - End
```

Let's break down and understand this second example line by line.

Line 58:

```
58: from parla import Parla
```

This imports the Parla runtime, for which the program's execution will use to coordinate tasks.

Line 61:

```
61: from parla.cpu import cpu
```

This imports the 'cpu' declaration, which is used in "@spawn" decorators to designate a task for execution on the CPU.

Line 64:

```
64: from parla.tasks import spawn, TaskSpace
```

Line 64 imports `@spawn` decorator and `TaskSpace` class. A `TaskSpace` provides an abstract high-dimensional indexing space in which tasks can be placed, and a `TaskSpace` can be indexed using any hashable values and any number of indices. If a dimension is indexed with numbers, then those dimensions can be sliced. Note, however, that a `TaskSpace` is not a Python list and thusly cannot exploit the advanced-indexing features of Python. For example, `[-1]` does not refer to the last element on `TaskSpace`.  Also note that names such as `TaskSpace` and the names assigned to tasks (their taskIDs) are only used to express dependencies.

# Example 1

```
67 # Example 1 - Simple Task
68 def task_simple():
69     
70     # Declare the TaskSpace and call it 'SimpleTaskSpace'
71     taskSpace = TaskSpace('SimpleTaskSpace')
72 
73     # Spawn a simple task and assign it a taskid of 0 within taskSpace
74     @spawn(taskid=taskSpace[0])
75     def task_0():
76         print('\ttask_0')
77     
78     # Return the awaitable
79     return taskSpace[0]
```

Lines 67 - 79 reiterate the first tutorial's declaration of a simple task.  Line 71 declares a new TaskSpace and gives it a name of "SimpleTaskSpace" (as will be discussed shortly, this task does not have any dependencies and can be immediately schedule for execution).  Lines 75 and 76 declare the "task_0" function whose sole purpose is to print out a statement "\ttask_0".  Decorating this function with the "@spawn" decorator (as in line 74) declares this as a task that Parla will execute, and the task is assigned a taskid of "taskSpace[0]" (specifically, an index of "0" within the TaskSpace "taskSpace").  To reiterate, this task is immediately scheduled for execution!

Line 79 returns the task defined in lines 74 - 76 due to its usage as an "awaitable" object in line 191.

# Example 2

Moving on to example 2, we see that it is just like example 1, but this segment of code also spawns a second task.  Lines 94 - 96 define this second task just like the first example, 


```
93    # Spawn the second task, but only after taskSpace_2[0] has completed
94    @spawn(taskid=taskSpace_2[1], dependencies=[taskSpace_2[0]])
95    def task_1():
96        print('\ttask_1')
```

Line 94 spawns a single task assigned an index 1 of the TaskSpace `taskSpace_2`, and the second parameter of the `@spawn` decorator specifies which tasks serve as the dependencies for this particular task. If dependencies are specified, the task will be scheduled and ran after all previous tasks have completed. In this case, the `task_1()` task has been specified as a successor of `task_0()`, and `task_1()` will execute after `task_0()` completes.

# Example 3

The `dependencies` parameter of the `@spawn` decorator can be a list of any combination of tasks and collections of tasks. Let's look at the third example at line 103, `task_spawn_tasks_in_a_loop()`, to see how task lists are specified at the `dependencies` keyword argument.

```
102 # Example 3 - Spawning tasks in a loop
103 def task_spawn_tasks_in_a_loop():
104 
105     # Constant for number of tasks to spawn
106     NUMBER_OF_TASKS_TO_SPAWN = 5
107 
108     # Declare a task space and call it 'LoopTaskSpace'
109     taskSpace_3 = TaskSpace('LoopTaskSpace')
110 
111     # Loop to instantiate tasks
112     for i in range(NUMBER_OF_TASKS_TO_SPAWN):
113 
114         # Spawn a task assigning it a taskid within taskSpace_3, but
115         # only after the previous dependencies have completed
116         @spawn(taskid=taskSpace_3[i], dependencies=[taskSpace_3[:i]])
117         def general_task():
118             print('\tgeneral_task_',i, sep='')
119
120     # Return the last task for the call to "await"
121     return taskSpace_3[NUMBER_OF_TASKS_TO_SPAWN - 1]
```

In line 116, TaskSpace `taskSpace_3` is sliced by the current index up to the current tasks, and this produces a series of tasks.  For the first iteration, this means that the first task that is spawned will not have dependences, but for the last iteration, the spawned task will have dependencies `taskSpace_3[0], taskSpace_3[1], taskSpace_3[2], and taskSpace_3[3]`. This causes each task to have each previous task as a dependency and will not start until those dependencies complete.  Therefore, the output will print taskids in increasing order.

Shown below is sample output from Example 3

```
Task Spawning Tasks In A Loop - Start
        general_task_0
        general_task_1
        general_task_2
        general_task_3
        general_task_4
Task Spawning Tasks In A Loop - End
```

# Example 3a

```
124 # Example 3a - Spawning tasks in a loop without specifying the dependencies
125 def task_spawn_tasks_in_a_loop_without_dependencies():
126 
127     # Constant for number of tasks to spawn
128     NUMBER_OF_TASKS_TO_SPAWN = 5
129 
130     # Declare a task space and call it 'LoopTaskSpace'
131     taskSpace_4 = TaskSpace('LoopTaskSpace2')
132 
133     # Loop to instantiate tasks
134     for i in range(NUMBER_OF_TASKS_TO_SPAWN):
135 
136         # Spawn a task assigning it a taskid within taskSpace_4
137         # NOTE: Contrasting with example 3, each task is spawned
138         # without specifying dependencies
139         @spawn(taskid=taskSpace_4[i])
140         def general_independent_task():
141 
142             # Give the 'later' tasks a smaller problem
143             # size to simulate 'future' tasks completing
144             # before previously-spawned tasks
145             iterations = 10000000 - pow(10, i + 2)
146 
147             # Output statement
148             print('\tgeneral_independent_task_', i, ' with ', iterations, ' iterations', sep='')
149 
150             # Simulate busy work
151             for j in range(iterations):
152                 pass
153 
154     # Return the last task for the call to "await"
155     return taskSpace_4[NUMBER_OF_TASKS_TO_SPAWN - 1]
```

Example 3a is a counter-example to that of example 3, where it can be seen that tasks specified without dependencies will not necessarily be executed in order. Shown below is a sample output of lines 124 - 155.

# Example 4

```
158 # Example 4 - Spawning tasks only after a previous taskSpace's tasks have completed
159 def task_different_taskSpace_dependency():
160 
161     # Declare a task space and call it 'FirstTaskSpace'
162     first_taskSpace = TaskSpace('FirstTaskSpace')
163 
164     # Spawn a task in the first task space
165     @spawn(first_taskSpace)
166     def task_from_first_taskSpace():
167         print('\tTask from first taskSpace')
168 
169     # Declare a second task space and call it 'SecondTaskSpace'
170     second_taskSpace = TaskSpace('SecondTaskSpace')
171 
172     # Spawn a task in the second task space, but only after
173     # the tasks in the first taskSpace have completed
174     @spawn(second_taskSpace, [first_taskSpace])
175     def task_from_second_taskSpace():
176         print('\tTask from second taskSpace')
177     
178     # Return the awaitable second_taskSpace
179     return second_taskSpace
```

The last example (lines from 158 to 179) shows that Parla can mix tasks for different TaskSpaces as dependencies. In addition, Parla could specify relationships between the tasks themselves that have been placed in different TaskSpaces. The first task is placed in the TaskSpace "first_taskSpace" (line 162). The second task is placed in the TaskSpace "second_taskSpace" (line 170). This task is specified as being dependent on the first task through the second argument `[first_taskSpace]` (note that the "dependencies=" keyword argument has been omitted).  This means that this second task will wait on the queue until the first task completes.  Line 179 returns the second_taskSpace as the awaitable object in the main function.

Parla allows users to use task features regardless of the TaskSpace they are placed in. Shown below is sample output from Example 4:

```
Task taskSpace dependency - Start
        Task from first taskSpace
        Task from second taskSpace
Task taskSpace dependency - End
```

# Further

In Parla, all tasks, task spaces, and task sets are awaitable. This is useful when you want to block until previous tasks have completed. As the last lesson, let's see how Parla exploits `await` syntax of Python.

```
182 # Define the main function (required of all Parla implementations)
183 def main():
184 
185     # Spawn a task, 'placing' it on the cpu
186     @spawn(placement=cpu)
187     async def start_tasks():
188 
189         # Test Example 1 
190         print('Task - Start')
191         await task_simple() # Wait for the task to complete
192         print('Task - End\n')
193 
194         # Test Example 2
195         print('Task Dependency - Start')
196         await task_simple_dependency() # Wait for the task to complete
197         print('Task Dependency - End\n')
198 
199         # Test Example 3
200         print('Task Spawning Tasks In A Loop - Start')
201         await task_spawn_tasks_in_a_loop() # Wait for the task to complete
202         print('Task Spawning Tasks In A Loop - End\n')
203 
204         # Test Example 3a
205         print('Task Spawning Tasks In A Loop Without Specifying Dependencies - Start')
206         await task_spawn_tasks_in_a_loop_without_dependencies() # Wait for the task to complete
207         print('Task Spawning Tasks In A Loop Without Specifying Dependencies - End\n')
208 
209         # Test Example 4
210         print('Task taskSpace dependency - Start')
211         await task_different_taskSpace_dependency() # Wait for the task to complete
212         print('Task taskSpace dependency - End\n')
```

Line 187 spawns an "entrance" task as an `async` function. This task calls and awaits the five functions as used by the above examples. Let's look what objects this function awaits.

```
79  return taskSpace[0]
...
99  return taskSpace_2[1]
...
121 return taskSpace_3[NUMBER_OF_TASKS_TO_SPAWN - 1]
...
155 return taskSpace_4[NUMBER_OF_TASKS_TO_SPAWN - 1]
...
179 return second_taskSpace
```

The first function call at line 191, `task_simple()` returns `taskSpace[0]` at line 79. The second function at line 196, `task_simple_dependency()` returns `taskSpace_2[1]` at line 99. This means that line 191 will not proceed until `task_simple()` completes, and this also means that line 196 will not proceed until `task_simple_dependency()` completes.  This applies for each of the subsequent awaits.

Congratulations! You've learned about the `TaskSpace` class, the specification for task dependencies, and about the awaitability of Parla.