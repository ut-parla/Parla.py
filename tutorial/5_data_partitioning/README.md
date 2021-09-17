# Lesson 5: Data Partitioning

Parla provides partitioners to partition 1-D or 2-D arrays into "logical devices". These partitions can be moved around different (physical) devices *automatically* on demand by the Parla runtime.

This section introduces how to use the Parla partitioners and the power of automatic data movement.

The provided script takes an 1-D array as an example.
```
data = np.array([0, 0, 0, 1, 2, 3])
```
The purpose of the subsequent tasks is to overwrite the first half of the array with the second half. To do so, we first create a Parla partitioner (aka mapper), `LDeviceSequenceBlocked`. For 2-D partitioning, Parla provides partitioners of two schemes, namely `LDeviceGridBlocked` and `LDeviceGridRaveled`.
```
mapper = LDeviceSequenceBlocked(2, placement=cpu)
partitioned_view = mapper.partition_tensor(data)
```
We initialize the partitioner such that it partitions the array into 2 halves. The initial placement of the partitions is on CPU.

Now, we are going to 1) perform the overwrite on GPU and 2) check the result on CPU. We take these two steps in two different tasks, one depending on the other.
```
@spawn(task[0], placement=gpu)
def t1():
    print("on GPU, data is automatically moved here as", type(partitioned_view[0]))
    print("overwrite the first half with the second half")
    partitioned_view[0] = partitioned_view[1]
```
The first task specifies the placement as `gpu`. Although you don't explicitly do any data movement operation, you'll find that the data partitions are automatically moved to the GPU as `cupy._core.core.ndarray`. Reading from the second half automatically pulls the data from CPU and writing to the first half will be pushed back to CPU in a lazy manner. The second task will verify if the writing is successful.
```
@spawn(task[1], dependencies=[task[0]], placement=cpu)
def t2():
    print("on CPU, data is automatically moved here as", type(partitioned_view[0]))
    print("data at the end:", data)
```
We specified dependencies for the second task so that it happens after the writing is done. The output will show the result with the overwritten array as `numpy.ndarray` on CPU.
