# Lesson 3: Devices and Architectures

Parla provides features to exploit heterogeneous architectures. Each task is allowed to
specify architectures or devices where it will be run.

This lesson introduces how to write heterogeneous tasks, and
function variants specialized to architectures, through a simple Parla example:
`devices_and_architectures.py`

This lesson needs four GPUs, CUDA, and CuPy.
You can run this example by the below command:

```
python devices_and_architectures.py
```

This script consists of examples showing various ways to spawn
and run tasks on different architectures with placement parameters.
The first example places a task on a CPU by specifying a CPU architecture to the placement.
The second example places a task on a GPU by specifying a GPU architecture to the placement.
The third example places a task on a specific GPU specified to the placement.
The fourth example shows how to spawn a task which could run either on a GPU or a CPU.
The fifth and sixth examples show how to spawn tasks in data-aware manner.
The last example shows how to exploit multiple GPU devices.

Parla allows users to omit the task placement.
If no architecture is specified, a task will be run on any device, which is available.

All example tasks carry out the same operations, element-wise vector addition, on
specified architecture/device, and print the outputs.
In this case, tasks placed on different architectures exploit function variants specialized
to corresponding to the target architecture/device.

Note that GPU tasks would require a bit of time
at the beginning since GPU tasks call and use a function variant exploiting
`cupy` library and it exploits
just-in-time compiler to convert Python functions to CUDA kernels.

This lesson skips explanations treated by the previous lessons.

Before introducing the examples,
let's first look at import statements from lines 1 to 7.

```
1  import cupy
2  import numpy
..
4  from parla.array import copy, clone_here
5  from parla.cpu import cpu
6  from parla.cuda import gpu
..
8  from parla.function_decorators import specialized
```

First, lines 5 and 6 import Parla packages used to specify placement of spawned tasks.
Spawned tasks placed on different architecture/device will utilize different
function variants.
To do this, line 1 and 2 import libraries of `cupy` for GPU and of `numpy` for CPU.
Line 8 imports Parla function decorators used to declare the specialized functions.
Line 4 imports functions of explicit data movement between CPU and GPU.

## Lesson 3-1: CPU Architecture

The first example shows a case that places a task on CPU architecture (Lines 53 to 59).

```
53  @spawn(placement=cpu)
```

Line 53 spawns a task, and places it on CPU architecture with `placement=cpu`.
If users specify an architecture, then a task will be run on any single device
with that architecture.
The term of `device` is different depending on architecture types in Parla.
In case of CPU, the device means one CPU core.
In case of GPU, the device means one CUDA stream.
In this example, this task will be run on the single core on any CPU.
There is a case that external libraries requires multiple cores.
In this case, users can set required environment values like
`OMP_NUM_THREADS` for the libraries.
However, the Parla runtime does not manage those resources.

This task calls `elemwise_add()` at line 58 and performs element-wise vector addition.
This is a simple task, and the function body declares and allocates input vectors.

```
12  @specialized
13  def elemwise_add():
```

You can see `@specialized` decorator of `elemwise_add()` at line 12.
It declares that `elemwise_add()` is specialized to a specific device, and
has a variant. In this script, line 23 declares a variant of `elemwise_add()` to be
run on GPU. We will visit this at the next lesson.

The below is outputs of the Lesson 3-1.

```
Spawns a CPU architecture task
CPU kernel is called..
Output>> 6 8 10 12
```

## Lesson 3-2: GPU Architecture

The second example places and runs a task on any GPU (Lines 65 to 69).
This example performs the same behavior, element-wise vector addition,
shown at the Lesson 3-1, but on GPU architecture.
Line 65 specifies that this task should be placed on GPU architecture.

```
65  @spwan(placement=gpu)
```

Parla supports a feature diverging execution path based on the current architecture,
called `variant`.
Let's look at line 23.

```
23  @elemwise_add.variant(gpu)
```

This decorator declares a function variant of `elemwise_add()`,
which is specialized to GPU tasks.
Calling the variant is transparent to users.
If GPU tasks call `elemwise_add()`, `elemwise_add_gpu()` is automatically called (Line 68).

`elemwise_add_gpu()` exploits `cupy` library to compute element-wise vector addition
on GPU (Lines 23 to 29).

Note that Parla does not enforce users to write device-specific codes at function variants.
For example, users still could call and use `numpy` library even
at a GPU specialized function.

The below is outputs of the Lesson 3-2.

```
Spawns a GPU architecture task
GPU kernel is called..
Output>> 6 8 10 12
```

## Lesson 3-3: GPU Device

The third example shows a case that assigns a specific GPU to a task
(Lines 75 to 79).

```
75  @spawn(placement=gpu(0))
```

Line 75 assigns `GPU0` to the task, instead of GPU architecture (Line 23).
This task is scheduled only on the GPU0 device.

The below is outputs of the lesson 3-3.

```
Spawns a single GPU task
GPU kernel is called..
Output>> 6 8 10 12
```

## Lesson 3-4: Heterogenous Task

This examples shows a task which could schedule either CPU or GPU architecture.

```
85  @spawn(placement=[cpu, gpu])
```

Line 85 passes both CPU and GPU to the placement parameter.
It means that this task could be scheduled on any architecture available first.
This task calls matching function variants to the target architecture.

## Lesson 3-5: Placing Task on CPU Through Data 

Parla allows to place tasks based on data locations.
If an array data is passed to placement,
Parla places the task to the device on which the data is allocated.
Note that the current Parla runtime supports only numpy and cupy arrays.
To show different placements depending on different data locations,
we declare two element-wise vector addition functions
accepting input array parameters. 

```
33  @specialzied
34  def elemwise_add_with_params(x, y):
...
41  @elemwise_add_with_params.variant(gpu)
42  def elemwise_add_with_params_gpu(x, y):
```

Line 34 declares a CPU function, and line 42 declares a GPU function variant.

This example places a task on CPU through numpy array data (Lines 94 to 102).

```
94  x_cpu = numpy.array([1, 2, 3, 4])
95  y_cpu = numpy.array([5, 6, 7, 8])
```

Lines 94 and 95 allocate two numpy arrays.
Both are loaded on CPU memory space.

```
96  @spawn(placement=[x_cpu, y_cpu])
```

Line 96 passes the numpy arrays to placement parameter.

```
101  elemwise_add_with_params(x_cpu, y_cpu)
```

Line 101 calls a function of element-wise vector addition.
Since this task is placed on CPU, it calls a CPU function at line 34.
The below is the output of this example.

```
Specifies a placement through data location
This should be running on CPU
Spawns a single task on CPU
CPU kernel is called..
Output>> 6 8 10 12
```

## Lesson 3-6: Placing Task on GPU Through Data 

The next example is to place a task on GPU through cupy arrays.

```
107  x_gpu = cupy.array([1, 2, 3, 4])
108  y_gpu = cupy.array([5, 6, 7, 8])
```

Lines 107 and 108 allocate two cupy arrays.
Both are loaded on GPU memory space.

```
109  @spaw(placement=[x_gpu, y_gpu])
```

Line 109 passes the cupy arrays to placement parameter.

```
114  elemwise_add_with_params(x_gpu, y_gpu)
```

Line 114 calls a function of element-wise vector addition as Lesosn 3-5 does.
However, since this task is placed on GPU, it calls a GPU function at line 42.
The below is the output of this example.

```
Specifies a placement through data location
Spawns a single task on GPU
This should be running on GPU
GPU kernel is called..
Output>> 6 8 10 12
```

## Lesson 3-7: Multi-GPU Tasks 

The last example shows typical programming patterns for multi-GPU tasks on Parla.
To fully exploit mulit-GPU, this example partitions input operands into
chunks as the number of GPUs, assigns each of chunk to GPU having the same index
to the chunk, and performs element-wise addition over the chunk vector on each GPU.

Those input operands are allocated as numpy arrays first,
and are copied and copied back to/from GPU through explicit data movement
of Parla.
Note that Parla also supports automatic data movement, but for simplicity,
this example sticks to the explicit data movement.

First, it declares a global variable, `NUM_GPUS`,
to set the number of GPUs to be used (Line 124).
You could update this number based on your system.

```
124  NUM_GPUS=4
```

Lines 126 to 128 declare and allocate operand and output numpy arrays.
All the vector size is 4. We will partition these vector into four scalar
chunks, and assign them to each GPU.

```
126  x = numpy.array([1, 2, 3, 4])
127  y = numpy.array([5, 6, 7, 8])
128  z = numpy.array([0, 0, 0, 0])
```

Now, line 132 and 134 spawn tasks as the number of GPUs, and
assigns each GPU ID to each task. For example, the task spawned
at the first iteration will be placed on GPU0.
Line 134 passes the assigned GPU ID to the task.

```
132  for gpu_id in range(NUM_GPUS):
133    @spawn(placement=gpu(gpu_id))
134    async def two_gpus_task(gpu_id=gpu_id):
```

Lines 137 and 138 copy the assigned vector chunk on CPU to the current GPU through
`clone_here()` of Parla. The current `clone_here()` implementation only supports
array parameter, not scalar parameter.

```
137  gpu_x = clone_here(x[gpu_id:(gpu_id+1)])
138  gpu_y = clone_here(y[gpu_id:(gpu_id+1)])
```

Then, line 139 performs add operator between `gpu_x` and `gpu_y` which were copied to
the current memory space.
The output is stored onto the local variable, `z_chunk`.

```
139  z_chunk = gpu_x + gpu_y
```

To update the output array on CPU, it copies back the `z_chunk` and updates
the assigned element of the output numpy array, `z[gpu_id]`, through
`copy()` (Line 140).

```
140  copy(z[gpu_id:(gpu_id+1)], z_chunk)
```

In the long run, the array `z` will have element-wise vector addition results
between `x` and `y` from all the four GPUs.

The below is outputs of this example.

```
GPU[0] calculates z[0]
GPU[1] calculates z[1]
GPU[2] calculates z[2]
GPU[3] calculates z[3]
Output>> 6 8 10 12
```

Congratulations! You've run how to utilize heterogeneous architectures/tasks on Parla.
