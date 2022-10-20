# Lesson 6: Automatic Data Movement

Instead of explicitly writing code to move data between devices, user could use Parla to manage data movement *automatically*. 

This section introduces PArray, a NumPy/CuPy-compatible array data structure provided by Parla. 

PArray is designed as a intelligent lightweight wrapper for NumPy and CuPy `ndarray`. Data wrapped by PArray will be automatically moved to the device where an access will happen.

The provided script show how to use PArray in a Parla program. At least two GPU devices are required to run this script.

### Create a PArray Obejct

To create a PArray, we need to initialize it with data.

This could be done by call `array` method with array like object (e.g. `list`, `ndarray`).

```Python
11    A = parray.array([[1, 2], [3, 4]])
```

We could also convert existing NumPy/CuPy array to PArray.

```Python
12    data = np.random.rand(2,2)
13    B = parray.asarray(data)
```

Now we have a PArray object that contains data in host memory.

### Manipulate with NumPy/CuPy API

PArray's support the same *member* methods as NumPy/CuPy `ndarray` does.

```Python
>> A = parray.array([[1, 2], [3, 4]])
>> A.argmax(axis=1)
[1 1]
```

*Static* methods of NumPy/CuPy could also be used with the help of two addition methods `.array` and `.update`.

`.array` returns a *view* of PArray, which has type `numpy.ndarray` (or `cupy.ndarray` for GPU tasks).

`.update` takes an ndarray as input and replaces the PArray's underlying buffer.

Examples:

```Python
B.update(cp.sqrt(B.array))
```

When only a portion of the PArray is assigned (slicing or indexing),  `.update`  could be omitted.

```Python
B[0][1] = np.sum((B + A).array)
```

### Task with PArray
To use a PArray object in a Parla Task, we need to put it in fields of `spawn`, so the scheduler will be able to know which array is used in the task and schedule a data movement in background.

There are three different access pattern:
#### Read Only
```Python
19        @spawn(ts[0], placement=gpu(0), input=[A])
20        async def read_only_task():
21            print(A)
```
If a task only does modify a PArray, it should be put in `input` field. And all read only operations are allowed here, (e.g `print`, copy).

`print` a PArray will give a dictionary, where key is the device index (`-1` for CPU device and `0, 1 ...` for GPU devices)

```Python
{0: array([[1, 2], [3, 4]]), 
1: None, 2: None, 3: None, 
-1: array([[1, 2], [3, 4]])}
```
Result of `print A` indicates that the array has been copied to GPU 0 automatically when this task starts.

#### Write Only

```Python
23      @spawn(ts[1], placement=gpu(1), output=[B])
24      async def write_only_task():
25          B.update(cp.sqrt(cp.random.rand(2, 2)))
```

If a task write to but didn't read from a PArray, it should be put in `output` field. And the PArray could be modified in this task.

#### Read and Write

```Python
27      @spawn(ts[2], [ts[0:2]], placement=cpu, input=[A], inout=[B])
28      async def write_and_write_task():
29          B[0][1] = np.sum((B + A).array)

```
PArray should be put in `inout` field if it is read and written by the same task.

In the above example, a task read from B's value and write to new value to B, while A is not modified. So A is put in `input` and B is put in `inout`.

### Fine-Grained Data Movement
Previous examples only show how PArray is moved as a whole. It is also useful to be able to move only a portion of it.

```Python
31      @spawn(ts[3], [ts[2]], placement=gpu(1), inout=[A[0]])
32      async def write_and_write_task():
33          A[0] = cp.sqrt(A[0].array)
```
Here only `A[0]` is moved.

Note: currently, fine-grained data movement on two overlapping portions is not supported.


### Memory Coherence

Since PArray will generate multiple copies of the same data in multiple devices, it needs to maintain the correct memory coherence in the system. `input`, `output` and `inout` not only are used by scheduler determine data dependence between tasks, but also used by PArray to update its coherence protocol. Therefore, it is necessary for programmer to correctly identify PArray in `input`, `output` and `inout`.

What's more, this memory model requires the application to be data race free in task granularity. That is, reads and writes on the same PArray between any two parallel tasks should be serilized.

For example, if there is no dependence between task A and task B, they may be scheduled to run in parallel. If A write on a PArray and B read from it, a data race will happen between two tasks and the behavior is undefined.

For fine grained data movement, read and write on two different portion of a PArray is not considered to be a race and could run correctly with parallel tasks. However, read and write on a portion and a complete copy will be considered as a race.
