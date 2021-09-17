# Lesson 5: Task Constraints

In this lesson we limit the resource usage of tasks and explore the benefits of doing so. 
There is only one source code file for this lesson: `task_constraints.py`  

The Parla `@spawn` decorator can take optional parameters limiting the resource usage of tasks. 
Two resource constraints are currently supported: `memory` and `vcus`. 
`vcus` are currently an experimental feature. 
In this lesson, we will explore usage of the `memory` constraint.  

In the current implementation of Parla, task constraints can best be thought of as hints to the Parla scheduler rather than as true restrictions. 
The `memory` constraint informs the Parla scheduler of the peak memory usage of a task in bytes, enabling it to make better scheduling decisions; 
however, Parla does not currently enforce the task to operate within its set constraints, and it is the programmer's duty to ensure that tasks do not exceed
the constraints passed to the scheduler.  

Let's take a look at how the `memory` constraint can be used to improve the scheduling of GPU-based tasks. 
When running GPU tasks, Parla creates a new CUDA stream in which to run each task. 
By default, though, it can only place one task on a GPU at a time. 
By expressing `memory` constraints, the programmer informs the Parla scheduler of the peak memory usage of a task, 
thereby enabling the scheduler to calculate how many tasks may be colocated on a single GPU. 
Thus, multiple tasks may run in separate streams on a single GPU. 
This technique has the advantage of overlapping communication and computation of different tasks, 
improving performance without the programmer having to work directly with CUDA.  

We now examine lines of interest for the code example provided in this lesson. 
In lines 10-16 we define a function which copies an array to the GPU, takes the square root of all elements, and then copies data back to the CPU. 
The `clone_here` function ensures that data is copied on the proper device and stream.  

```
10  def gpu_sqrt(data, i):
11      gpu_data = clone_here(data[i])
12      print(f"{cp.cuda.get_current_stream()}: data[{i}] host to device transfer complete")
13      cp.sqrt(gpu_data)
14      print(f"{cp.cuda.get_current_stream()}: data[{i}] computation complete")
15      data[i] = cp.asnumpy(gpu_data)
16      print(f"{cp.cuda.get_current_stream()}: data[{i}] device to host transfer complete")
```

In lines 27-31, we spawn two tasks, each performing `gpu_sqrt` on a large array of data on GPU 0. 
In these first two tasks, no memory constraints are set.  

```
27          print("\nStarting GPU tasks with no memory constraints")
28          for i in range(2):
29              @spawn(placement=gpu(0))
30              async def t1():
31                  gpu_sqrt(data, i)
```

After awaiting these two tasks, we spawn two more identical tasks, but this time with memory constraints set (in bytes).  

```
35          print("\nStarting GPU tasks with memory constraints")
36          for i in range(2):
37              @spawn(placement=gpu(0), memory=2**32)
38              async def t2():
39                  gpu_sqrt(data, i)
```

Here is an example output from running this script:

```
Initializing data on host

Starting GPU tasks with no memory constraints
<Stream 47606900461072>: data[0] host to device transfer complete
<Stream 47606900461072>: data[0] computation complete
<Stream 47606900461072>: data[0] device to host transfer complete
<Stream 47600451457104>: data[1] host to device transfer complete
<Stream 47600451457104>: data[1] computation complete
<Stream 47600451457104>: data[1] device to host transfer complete

Starting GPU tasks with memory constraints
<Stream 47606903095984>: data[0] host to device transfer complete
<Stream 47606903095984>: data[0] computation complete
<Stream 47600481891744>: data[1] host to device transfer complete
<Stream 47606903095984>: data[0] device to host transfer complete
<Stream 47600481891744>: data[1] computation complete
<Stream 47600481891744>: data[1] device to host transfer complete
```

Some observations:
- All four tasks ran on a unique CUDA stream.  
- The tasks without memory constraints ran in order on the GPU since Parla could only schedule one at a time.  
- The tasks with memory constraints overlap computation and communication for the two arrays.  
- You may not see this exact order when running code on your system, 
but importantly, you will never see interleaving of the tasks running in the first stage since only one can run at a time.  

As a final note, it is important to properly express the peak memory usage of your task if the scheduler is to properly colocate tasks on a single GPU. 
Take particular care to include extra memory used by out-of-place algorithms. 
If the `memory` constraint is lower than the actual peak memory usage, Parla may colocate too many tasks on a single GPU and run out of memory to allocate.  

Congratulations! You've learned to improve the performance of Parla applications by expressing task constraints!  
