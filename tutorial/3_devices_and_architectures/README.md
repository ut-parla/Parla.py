# Lesson 3: Devices and Architectures

Parla provides features to exploit heterogeneous architectures. Each task is allowed to
specify architectures or devices where it will be run.

This lesson introduces how to write heterogeneous tasks, and
function variants specialized to the specific architecuters, through a simple Parla example:
`devices_and_architectures.py`

You can run this example by the below command:

```
python devices_and_architectures.py
```

This script spawns two tasks and places them to CPU and GPU, respectively, performs
element-wise add oprations, and prints results.
In this case, the GPU task calls a function variant to GPU which exploits CuPy library.

The below is outputs of the example.

```
5 7 9
5 7 9
```

Let's break down and understand this script line by line.
In this lesson, we will skip lines explained by previous lessons.

First, lines 1, 4, and 6:

```
1  import cupy
..
4  from parla.cuda import gpu
..
6  from parla.function_decorators import specialized
```

Line 1 imports CuPy which GPU variant function would call.
Line 4 imports Parla GPU runtime.
LIne 6 imports a specialized decorator to exploit variant functions to architectures.


```
25  arch_modes = [cpu, gpu(0)]
26
27  for arch_mode in arch_modes:
28    @spawn(placement = arch_mode)
29    def elemwise_add_task()
```

Lines 27 to 29 spawn two tasks placed on architecutrs of CPU and GPU, respectively.
`@spawn(placement = cpu)` place a task on CPU, and `@spawn(placement = gpu(0))`
places a task on specifically GPU0 among GPUs. 
Note that Parla supports multiple architectures/devices to place a single task.
For simplicity, this example places tasks on the single architecture/device.

The task, `elemwise_add_task()`, calls specialized functions doing element-wise addition 
between vectors. Let's take a look how to specialize functions.

```
 9  @specialized
10  def elemwise_add():
```

In line 9, the decorator of `@specialized` declares the function, `elemwise_add()`, has
specialized functions. Depending on the target architecture of the caller task, a proper
variant function is called.

```
16  @elemwise_add.variant(gpu)
17  def elemwise_add_gpu():
..
20    cupy_elemwise_add = cupy.ElementwiseKernel(...)
```

Line 16 declares a function variant for GPU through the decorator of `@element_add.variant(gpu)`.
This function runs on GPU and utilizes CuPy for element-wise vector addition (Line 20).

```
30  elemwise_add()
```

Lines 30 calls `element_wise()`.
Specialized functions are hidden from users and Parla
automatically calls an appropriate function based on an architecture/devices of the current
task.

Congratulations! You've run how to utilize heterogeneous architectures/tasks on Parla.
