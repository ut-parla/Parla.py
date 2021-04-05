# Blocked QR Factorization Example
**Various implementations of a blocked QR factorization to test the performance of various features.**

### Blocked QR Factorization
Given a matrix A, a [QR factorization](https://en.wikipedia.org/wiki/QR_decomposition) is the decomposition
of the matrix A into the matrix product A = QR where Q
is comprised entirely of orthogonal unit vectors and R is an
upper triangular matrix. It's a candidate for testing nested parallelism because blocks can be processed in parallel, and elements within blocks can be processed in parallel. For more information see "[Direct QR factorizations for tall-and-skinny matrices in
MapReduce architectures](https://arxiv.org/abs/1301.1071)."

### Setup
You'll need NumPy, CuPy, Dask, and of course Parla in your conda enviroment for these to run. For the first three, you can just do
```
conda install numpy cupy dask
```
For Parla, I just do
```
export PYTHONPATH="/path/to/Parla.py" # Replace with your path
```
Some of these programs use VECs (Virtual Execution Contexts), which require additional setup.

### Usage
The main program with a Parla implementation of the blocked TSQR algorithm is qr_parla.py. It can simply be run with
```
python qr_parla.py
```
For information on the program, run
```
python qr_parla.py -h
```
An example run would be
```
qr_parla.py -r 500000 -c 1000 -b 125000 -i 1 -g 4 -p gpu # Factorize a 500k by 1k matrix broken into 125k-row blocks on 4 GPUs, 1 iteration
```

### Useful Files
- `README.md`
	- This
- `qr_parla.py`
	- The main script implementing a TSQR factorization with Parla; usage is described above.
- `my_sbatch.bash`
	- Helper script I (Sean) like to use for submitting jobs on Frontera
- `experiment.sbatch`
	- Slurm job that I submit on Frontera. **Great place to look for examples on how to run `qr_parla.py`**
- `qr_numpy.py`
	- Tests NumPy's basic factorization algorithm for CPUs, `numpy.linalg.qr()`
- `qr_cupy.py`
- 	- Tests CuPy's basic factorization algorithm for GPUs, `cupy.linalg.qr()`
- `qr_dask.py`
	- Tests Dask's blocked version for CPUs, `dask.linalg.qr()`

#### Other Files
- `qr_simple_blocked.py`
	- Original proof of concept for blocked implementation. Doesn't actually parallelize over blocks. Not used for testing.
- `qr_parla_maybe_segfault.py`
	- Reproduces a segfault that occurs inconsistently; further testing needed.
- `qr-multithread.py`
	- Previously used for testing performance of VECs. Attempt to achieve nested parallelism by spawning multiple threads. Doesn't actually work, as all NumPy calls are multiplexed onto a single group of threads (obviating the need for VECs to manage different contexts if parallelism is to be achieved in Python with threads.)
- `qr-multiprocess.py`
	- Previously used for testing performance of VECs. Achieves nested parallelism by spawning multiple processes, requiring data to be copied.
- `qr-vec.py` \*
	- Uses VECs (Virtual Execution Contexts) in order to achieve nested parallelism using multiple threads in a single virtual address space.
- `vec-segfault.py` \*
	- Slimmed down version of qr-vec.py for isolating a segmentation fault that has yet to be fixed.  
	- 
\* Note that VECs have their own special installation process and don't work with vanilla Parla.
