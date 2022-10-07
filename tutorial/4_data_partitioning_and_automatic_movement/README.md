# Lesson 4: Data Partitioning and Automatic Movement

Parla provides partitioners to partition 1-D or 2-D arrays into `logical devices`.
These partitions can be moved around different (physical) devices *automatically*
on demand by the Parla runtime.

This section introduces how to use the Parla partitioners and the automatic data movement.

The provided script starts from an 1-D random array of length 4 for demonstration.

```
data = np.random.rand(4)
```

This section presents the following three examples:

1. Evenly partition the initial array into two disjointed arrays, placing each to CPU and GPU0, respectively.
2. Add up the two partitions on GPU0 without explicit data movement.
3. Same as the step 2 on CPU.

For step 1, we first create a Parla partitioner (aka mapper), `LDeviceSequenceBlocked`.
(For 2-D partitioning, Parla provides partitioners of two schemes, namely `LDeviceGridBlocked` and `LDeviceGridRaveled`.)
```
mapper = LDeviceSequenceBlocked(2, placement=[cpu[0], gpu[0]])
partitioned_view = mapper.partition_tensor(data)
```

We print out the details (value, type, and residence device) of the partitions.

Next, we perform the addition on GPU0.
```
@spawn(task[0], placement=gpu[0])
def t1():
    sum_on_gpu = partitioned_view[0] + partitioned_view[1]
```
Although we don't explicitly do any data movement operation like `clone_here`,
the input partitions are automatically moved to GPU0 as a cupy array.
The output is also the cupy array allocated on the same device.
The subsequent print statements show the processes explained above.
Similarly, without any explicit data movement, we could perform the addition on CPU.
```
@spawn(task[1], dependencies=[task[0]], placement=cpu[0])
def t2():
    sum_on_cpu = partitioned_view[0] + partitioned_view[1]
```
All the inputs and the output are numpy arrays on CPU.

Example output (values may vary due to random generation):
```
Initial array:  [0.0603186  0.11194458 0.99213128 0.20585806]
=======
Partitions:
[0.0603186  0.11194458] of type <class 'numpy.ndarray'> on device <CPU 0>
[0.99213128 0.20585806] of type <class 'cupy._core.core.ndarray'> on device <CUDA 0>
=======
On GPU, inputs are automatically cloned here
input types: [<class 'cupy._core.core.ndarray'>, <class 'cupy._core.core.ndarray'>]
output type: <class 'cupy._core.core.ndarray'>
output: [1.05244989 0.31780264]
=======
On CPU, inputs are automatically cloned here
input types: [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>]
output type: <class 'numpy.ndarray'>
output: [1.05244989 0.31780264]
```
