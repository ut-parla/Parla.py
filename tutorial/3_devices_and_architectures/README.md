# Lesson 3: Devices and Architectures

Parla provides features to exploit heterogeneous architectures.
In Parla, users are able to select a specific device or a specific type
of architecture for each task to launch.

This lesson introduces task placements in Parla through the following simple examples:
`cpu.py` `gpu.py` `hetero_devices.py`

These examples need four GPU devices, CUDA, and CuPy packages.
For this example, we use an element-wise vector addition operation as the main
computation of a task.

For simplicity, we will not address features that we handled in the previous
examples.

## Lesson 3-1: CPU tasks

The first example places a task on a CPU.
You can run this example by the below command:
```
python cpu.py
```

Now, let's understand how to place a task on CPU.
We need to import a Parla cpu package (Line 3). 

```
1  import numpy
2  from parla import Parla
3  from parla.cpu import cpu
4  from parla.tasks import spawn, TaskSpace
```

Line 16 and 17 declares and initialize numpy array objects which are
operands of the element-wise vector addition. Line 22 sets a placement
of a task, `cpu_arch_task`, to a CPU architecture and hence, this task will run on
a single CPU core.

```
22  @spawn(placement=cpu)
```

Parla allows to specify task placements through Numpy objects.
Line 30 passes two numpy arrays, `x` and `y`, to the `placement` argument. 
This task will be launched to any available one of devices where `x` or `y` was instantiated 
when this task was spawned. In this example, this task must run on CPU since
`x` and `y` are numpy arrays located on CPU.

```
30  @spawn(placement=[x, y])
```

The below is an output of the `cpu.py`.

```
Spawns a CPU architecture task.
CPU kernel is called..
Output>> 6 8 10 12

Specifies a placement through data location
This should be running on CPU
Spawns a single task on CPU
CPU kernel is called..
Output>> 6 8 10 12
```

## Lesson 3-2: GPU tasks

Now, let's learn how to create GPU tasks. 
You can run this example by the below command:

```
python gpu.py
```

This lesson consists of two examples. The first example spawns a
task on a single GPU device and performs element-wise vector addition.
The next example exploits advanced features to leverage multi-devices
heterogeneous architecture. This example partitions each input array into 4 slices,
distributes each of them to respective GPU devices, and performs element-wise vector addition
between the slices.

Note that GPU tasks would require a bit of time in the beginning for
just-in-time compilation for `cupy` libraries.

In this example, we need to import cupy (Line 1) and a Parla gpu (Line 6) packages. 
```
1  import cupy
...
6  from parla.cuda import gpu
```

Let's start from the first example from line 23 to 34.
Line 23 sets a placement of `gpuarch_place_case` task to a GPU architecture.
```
23  @spawn(placement=[gpu])
```
Therefore, this task will be run on any avilable single GPU device.
Parla allows application programmers to call any Python external libraries
without any code modification. Based on that, line 26 and 27 declare necessary cupy
arrays, and line 31 declares and calls a cupy kernel for element-wise vector
addition. 

```
12  def elemwise_add(x, y):
13    print("GPU kernel is called..")
14    cupy_elemwise_add = cupy.ElementwiseKernel("int64 x, int64 y", "int64 z", "z = x + y")
15    return cupy_elemwise_add(x, y)
...
31  z_g = elemwise_add(x_g, y_g)
```

Now, let's move to the next example from lines 40 to 58. This example uses
four numpy arrays, `x_c`, `y_c`, and `z_c`, declared from lines 42 to 44.
Parla allows selecting a specific device for a task. Lines 48 to 50 spawns a single
task for each GPU device.

```
48  for gpu_id in range(NUM_GPUS):
49    @spawn(placement=gpu(gpu_id))
50      async def workpart_across_gpus(gpu_id=gpu_id):
```

Each task takes a slice of `x_c`, `y_c`, and `z_c`, and performs element-wise vector
addition between them. Since all data are numpy arrays and are located on CPU memory, 
each task first should copy the slices to the current GPU device memory.
To simplify data movements between devices, Parla provides the following APIs: `clone_here()`
and `copy()`. `clone_here()` copies a data to the current device memory regardless
of the data's location. `copy()` copies data between parameter arrays
regardless of their current locations. By exploiting them, lines 53 to 56 copies data
to the current device, calculates elementwise addition, and copies back to `z_c`
that is located on CPU memory.

```
53  tmp_x = clone_here(x_c[gpu_id:(gpu_id+1)])
54  tmp_y = clone_here(y_c[gpu_id:(gpu_id+1)])
55  z_chunk = elemwise_add(tmp_x, tmp_y)
56  copy(z_c[gpu_id:(gpu_id+1)], z_chunk)
```

The below is an output of `gpu.py`.

```
This should be running on GPU
GPU kernel is called..
Output>> [ 6  8 10 12] 

GPU[0] calculates z[0]
GPU kernel is called..
GPU[1] calculates z[1]
GPU kernel is called..
GPU[2] calculates z[2]
GPU kernel is called..
GPU[3] calculates z[3]
GPU kernel is called..
Output>> 6 8 10 12
```

## Lesson 3-2: Heterogeneous Task 

This examples shows a task which could schedule either CPU or GPU architecture.
You can run this example by the below command:

```
python hetero_devices.py
```
Now, let's see how to implement a heterogeneous task in Parla. 
Line 35 sets the placement to a CPU or GPU architecture.

```
35  @spawn(placement=[cpu, gpu])
```
Therefore, when this task is launched, it will run either a CPU core or a GPU device.
Different types of devices should call different kernel codes. 
To handle this case, Parla supports `@specialized` and
`@[ORIGINAL FUNCTION].variant([ARCHITECTURE TYPE])` 
decorators. If `@specialized` is prepended to a function declaration, it implies that
that function is for a CPU execution and its variants for different computing devices may
exist in the program.
Line 11 declares a CPU element-wise vector addition, and sets that its variant may exist. 
```
11  @specialized
12  def elemwise_add():
```
Line 20 declares a variant of the `elemewise_add()` that exploits cupy kernels
for a GPU execution.

```
20  @elemwise_add.variant(gpu)
21  def elemwise_add_gpu():
```
When the `elemwise_add()` is called at line 38, the Parla runtime automatically finds
the placement of `single_task_on_both` task, finds a compatible function variant, and
calls it.  

The below is an output of `hetero_devices.py`.

```
Spawns a single task on either CPU or GPU
GPU kernel is called..
Output>> 6 8 10 12 
```

Congratulations! You've run how to utilize heterogeneous architectures/tasks on Parla.
