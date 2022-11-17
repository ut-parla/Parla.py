# Lesson 3: Devices and Architectures

Parla provides features to organize execution on heterogeneous architectures. We currently only support Nvidia GPUs, but have active development on AMD GPU support and plan to extend to other device types and accelerators in the future.

When launching a task, the user can list a `placement` constraint to specify a specific device or type of device that the task can launch on.

This lesson introduces task placements in Parla through the following simple examples:
`cpu.py`, `gpu.py`, and `hetero_devices.py`

To run these examples, you need at least 2 GPU devices, a CUDA driver & runtime, and CuPy.

We use an element-wise vector addition operation as the main
computation of a task.

## Device Initialization

In the first three lessons you've already been running tasks on the CPU. By default, Parla schedules an unlabeled task on any of the initialized device. So far the lessons have only initialized CPU devices through the following import:

```python
from parla.cpu import cpu
```

This configures and adds a "CPU" device to the runtime using your system's information. The `cpu` object is the general class of "CPU" type devices.

If you want to activate all CUDA devices on the machine, you can import the `cuda` module. Note that this requires a CUDA runtime and CuPy.

```python
from parla.cuda import gpu
```

This will initialize all devices seen by the `CUDA_VISIBLE_DEVICES` environment variable.
The `gpu` object represents the general class of "CUDA GPU" type devices. Querying `gpu(i)` returns the specific `i`th device.

All devices your application uses must be initialized before the Parla runtime context manager is entered.

## Task Placement

The spawn decorator takes a `placement` argument. Placement takes a device type, device, ndarray, or task.

If the argument is a device type, then the task may be scheduled on any available device of the specified type.
If it is a specific device, the task will only be scheduled on that device.
If it is an ndarray (cupy or numpy), the task will be scheduled on the device that holds that data at spawn time.
If the placement argument is a task, then the spawning task will be scheduled on the same device as the argument task.

For example, the following will constrain a task to only execute on a gpu.

```
@spawn(placement=gpu)
```

And the following will constrain a task to only execute on the `0`th gpu.

```
@spawn(placement=gpu(0))
```

Placement can also take a list of objects. In this case, the task could be scheduled on any of the specified devices.

Specify task placements through `ndarray` objects.
This task will be launched to either of devices where `x` or `y` was defined
when this task was spawned.

```
x = np.ones(10)
with cp.cuda.Device(1):
    y = cp.ones(10)
@spawn(placement=[x, y])
def task():
    //do work
```

The below is an output of `cpu.py`.

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

## Relative Data Movement

In `gpu.py`, all data is initialized on the host. Each task must first should copy their corresponding slice to device memory.
To simplify data movements between devices at runtime, Parla provides the following APIs: `clone_here()` and `copy()`. `clone_here()` copies a data to the current device memory regardless of the data's location. `copy()` copies data between arrays
regardless of their current locations.

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

## Heterogeneous Function Variants

Some tasks can be scheduled on either CPU or GPU architectures.
You can find an example in `hetero_devices.py`.

Line 35 sets the placement to a CPU or GPU architecture.

```
35  @spawn(placement=[cpu, gpu])
```

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

Parla defines variants of functions instead of tasks. This allows the construction of larger tasks with bodies is composed of multiple function variants. The modularity allows parts of a task to be ported to the device slowly.

The below is an output of `hetero_devices.py`.

```
Spawns a single task on either CPU or GPU
GPU kernel is called..
Output>> 6 8 10 12
```

Congratulations! You've learned about how to utilize heterogeneous architectures/tasks in Parla.
