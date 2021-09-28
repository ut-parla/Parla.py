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

This script consists of examples based on target architectures and devices.
The first example introduces how to place a task on a CPU architecture.
The second example introduces how to place a task on a GPU architecture.
The third example introduces how to place a task on a specfied GPU device.
The fourth example introduces how to exploit multiple GPU devices.

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
4  from parla.cpu import cpu
5  from parla.cuda import gpu
..
7  from parla.function_decorators import specialized
```

First, lines 4 and 5 import Parla packages used to specify placement of spawned tasks.
Spawned tasks placed on different architecture/device will utilize different
function variants.
To do this, line 1 and 2 import libraries of `cupy` for GPU and of `numpy` for CPU.
Line 7 imports Parla function decorators used to declare the specialized functions.

## Lesson 3-1: CPU architecture

The first example shows a case that places a task on CPU architecture (Lines 35 to 41).

```
35  @spawn(placement=cpu)
```

Line 35 spawns a task, and places it on CPU architecture with `placement=cpu`.
If users specify an architecture, then a task will be run on any single device
with that architecture.
The term of `device` is different depending on architecture types in Parla.
In terms of CPU, the device means cores. Other devices like GPU and FPGA mean a sinlge
GPU and FPGA.
Therefore, in this example, this task will be run on any cores of any CPU.
The number of CPU cores used by Parla is set through existing CPU libraries, like
`OMP_NUM_THREADS`.

This task calls `elemwise_add()` at line 40 and performs element-wise vector addition.

```
12  @specialized
13  def elemwise_add():
```

You can see `@specialized` decorator of `elemwise_add()` at line 12.
It declares that `elemwise_add()` is specialized to a specific device, and
has a variant. In this script, line 23 declares a variant of `elemwise_add()` to be
run on GPU. We will revisit this at the next lesson.

The below is outputs of the Lesson 3-1.

```
Spawns a CPU architecture task
6 8 10 12
```

## Lesson 3-2: GPU architecture

The second example shows a case that places a task on GPU architecture (Lines 47 to 51).
This example does the same behavior, element-wise vector addition, shown at the Lesson 3-1.
The main difference is this task is placed on GPU architecture (Line 47).
Therefore, this task will be run on any single GPU device.

```
47  @spwan(placement=gpu)
```

Parla supports a feature diverging execution path based on the current architecture,
called `variant`.
Let's look at line 22.

```
22  @elemwise_add.variant(gpu)
```

This decorator declares that this function is a variant of the function `elemwise_add()`
and is specialized to tasks placed on GPU.
So to speak, this function, `elemwise_add_gpu()` is called
when tasks placed on GPU call `elemwise_add()` (line 50).

`elemwise_add_gpu()` exploits `cupy` library to perform element-wise vector addition
on GPU (lines 22 to 27).

Note that Parla does not enforce users to use device-specific codes at function variants.
For example, users still could call and use `numpy` library even though that function
is specialized to GPU.

The below is outputs of the Lesson 3-2.

```
Spawns a GPU architecture task
6 8 10 12
```

## Lesson 3-3: GPU device

The third example shows a case that points a specific GPU for a task placement
(lines 57 to 61).

```
57  @spawn(placement=gpu(0))
```

Line 57 sets the placement of the task to `GPU0`, instead of GPU architecture
as line 22 does.
Therefore, this task will be run on only GPU0 device.

The below is outputs of the lesson 3-3.

```
Spawns a single GPU task
6 8 10 12
```

## Lesson 3-4:

The last example shows a simple case that partitions input operands into
chunks as the number of GPUs, assigns each of them to GPU having the corresponding
chunk index, and performs element-wise chunk vector addition on each GPU.

First, we use global variable, `NUM_GPUS`, to set the number of GPUs to be used.
If you do not have four GPUs, you could decrease this number.

```
70  NUM_GPUS=4
```

Lines 72 to 74 declare operand and output vectors.
The vector size is all 4.

```
72  x = cupy.array([1, 2, 3, 4])
73  y = cupy.array([5, 6, 7, 8])
74  z = numpy.array([0, 0, 0, 0])
```

Next, lines 77 to 80 partition the vectors into four chunks.
Therefore, each chunk size is 1, which is a scalar value.

```
77  mapper = LDeviceSequenceBlocked(NUM_GPUS, placement=cpu)
78  x_view = mapper.partition_tensor(x)
79  y_view = mapper.partition_tensor(y)
80  z_view = mapper.partition_tensor(z)
```

Now, line 84 and 85 spawns tasks as the number of GPUs, and
assigns each GPU ID to each task. For example, the task spawned
at the first iteration will be placed on GPU0.
Line 86 passes its assigned GPU ID to task.
This implies that chunks having the same index with assigned GPU
is assigned to and is computed through this task.

```
84  for gpu_id in range(NUM_GPUS):
85    @spawn(placement=gpu(gpu_id))
86    async def two_gpus_task(gpu_id=gpu_id):
```

Then, line 89 performs add operation between assigned chunks of the operands,
and produces partial result.

```
89    z_view[gpu_id] = x_view[gpu_id] + y_view[gpu_id]
```

Therefore, the complete addition result is distributed and computed on four GPUs. 
Parla could exploit multiple GPUs in various ways to relieve memory and computation
loads.
The below is outputs of this example.

```
GPU[0] calculates z[0]
GPU[1] calculates z[1]
GPU[2] calculates z[2]
GPU[3] calculates z[3]
6 8 10 12
```

Congratulations! You've run how to utilize heterogeneous architectures/tasks on Parla.
