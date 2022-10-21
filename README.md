```
______          _                 ┌─────────┐┌──────┐
| ___ \        | |                │ Task A  ││Task B│
| |_/ /_ _ _ __| | __ _           └┬───────┬┘└────┬─┘
|  __/ _` | '__| |/ _` |          ┌▽─────┐┌▽─────┐│  
| | | (_| | |  | | (_| |          │Task D││Task C││  
\_|  \__,_|_|  |_|\__,_|          └┬─────┘└┬─────┘│  
                                  ┌▽─────┐┌▽──────▽┐ 
                                  └──────┘└────────┘ 
```

# Introduction

**Parla** is a task-parallel programming library for Python.
Parla targets the orchestration of heterogeneous (CPU+GPU) workloads on a single shared-memory machine.
We provide features for resource management, task variants, and automated scheduling of data movement between devices. 

We design for *gradual-adoption* allowing users to easily port sequential code for parallel execution.

The Parla runtime is multi-threaded but *single-process* to utilize a shared address space. 
In practice this means that the main compute workload within each task *must* release the CPython Global Interpreter Lock (GIL) to achieve parallel speedup.

Note: Parla is not designed with workflow management in mind and does not currently support features for fault-tolerance or checkpointing.

# Installation

Parla is currently distributed from this repository as a Python module.

Parla 0.2 requires `Python>=3.7`, `numpy`, `cupy`, and `psutil` and can be installed as follows:

```bash
conda (or pip) install -c conda-forge numpy cupy psutil
git clone https://github.com/ut-parla/Parla.py.git
cd Parla.py
pip install .
```
To test your installation, try running

```bash
python tutorial/0_hello_world/hello.py
```

This should print

```bash
Hello, World!
```

We recommend working through the tutorial as a starting point for learning Parla!

## Example Usage

Parla tasks are launched in an indexed namespace (the '*TaskSpace*') and capture variables from the local scope through the task body's closure.

Basic usage can be seen below:

```python
with Parla:
    T = TaskSpace("Example Space")

    for i in range(4):
        @spawn(T[i], placement=cpu)
        def tasks_A():
            print(f"We run first on the CPU. I am task {i}", flush=True)

    @spawn(T[4], dependencies=[T[0:4]], placement=gpu)
    def task_B():
        print("I run second on any GPU", flush=True)
```


## Example Mini-Apps
The examples have a wider set of dependencies.

Running all requires: `scipy`, `numba`, `pexpect`, `mkl`, `mkl-service`, and `Cython`.


To get the full set of examples (BLR, N-Body, and synthetic graphs) initialize the submodules:
```
git submodule update --init --recursive --remote
```

Specific running installation instructions for each of these submodules can be found in their directories.

The test-suite over them (reproducing the results in the SC'22 Paper) can be launched as:

```bash
python examples/launcher.py --figures <list of figures to reproduce>
```

<!---
## Running the Docker Container
The Parla container requires CUDA support in the Docker host environment. To get a shell inside the provided docker container run

```
docker run --gpus all --rm -it utpecos/parla
```

In this container, a Parla repo with tutorial branch is put at the root of HOME directory, which could be used out of the box.

Depending on your Docker configuration, you may need to run this command as root using sudo or some other method. Since CUDA is required for all the demos, you must provide some GPUs for the docker container to use. For this to work using the command shown, you need to use Docker 19.03 or later.

## Virtual Execution Contexts (Experimental)

VECs are currently experimental.
Some of the packaging work for them still needs to be done.
Here are instructions for how to get the VEC prototype running locally:

Glibc is usually the pain point.
Everything should be working for numpy/openblas now.
Other libraries may or may not work.
Make sure you have recent versions of cython, numpy, gcc, binutils, make, libunwind, and CMake.
Everything except gcc is available via conda if you need it.
We recommend using conda-forge (see https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge), but the default packages may work too.
Leave "$HOME/Parla.py" empty since this will clone a bunch of stuff there.
Note: the current instructions reference build directories to get some compiled libraries at runtime.
We don't have a conda package set up for this yet.
We don't even really have nice installation and launching infrastructure either, so a bunch of the stuff here has some paths hard-coded until we get something better set up.
Copy the following into a script and run it (NOTE: this deletes the "$HOME/Parla.py" directory entirely each time, so DO NOT save stuff there and then use this script as-is to build):
```Shell
set -e
rm -rf "$HOME/Parla.py"
git clone https://github.com/ut-parla/Parla.py "$HOME/Parla.py"
cd "$HOME/Parla.py"
git clone https://github.com/ut-parla/glibc glibc
cd glibc
rm -rf install
mkdir install
rm -rf build
mkdir build
cd build
CC="gcc -no-pie -fno-PIE" CXX="g++ -no-pie -fno-PIE" MIG="mig" MAKE="make"\
 AUTOCONF=false MAKEINFO=: \
 ../configure \
 --host=x86_64-linux-gnu \
 --prefix="$HOME/Parla.py/glibc/install" \
 --enable-add-ons=libidn,"" \
 --enable-stackguard-randomization \
 --without-selinux \
 --enable-stack-protector=strong \
 --enable-obsolete-rpc \
 --enable-obsolete-nsl \
 --enable-kernel=3.2 --enable-multi-arch --enable-static-pie
make
make install
cd ../..
cd runtime_libs
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ../..
python setup.py build_ext --include-dirs="$CONDA_PREFIX/include" install;
cd ..
```
If that actually worked, you can now copy the following single line into a separate shell script (say parla.sh) and use that like the Python command.
NOTE: the interactive shell doesn't work on some machines right now and we don't know why yet, so only use this to run scripts.
```Shell
LD_LIBRARY_PATH="$HOME/Parla.py/runtime_libs/build:$LD_LIBRARY_PATH" LD_PRELOAD="$HOME/Parla.py/runtime_libs/build/libparla_supervisor.so" "$HOME/Parla.py/runtime_libs/usingldso" "$HOME/Parla.py/glibc/install" python "$@"
```
Using the parla.sh shell script you can run parla programs as "sh parla.sh $ARGS" where $ARGS is whatever arguments you'd be passing to Python.
-->


## Acknowledgements
This software is based upon work supported by the Department of Energy, National Nuclear Security Administration under Award Number DE-NA0003969.

## How to Cite Parla.py
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6941889.svg)](https://doi.org/10.5281/zenodo.6941889)

Please cite the following reference.

```
@inproceedings{
    author = {H. Lee, W. Ruys, Y. Yan, S. Stephens, B. You, H. Fingler, I. Henriksen, A. Peters, M. Burtscher, M. Gligoric, K. Schulz, K. Pingali, C. J. Rossbach, M. Erez, and G. Biros},
    title = {Parla: A Python Orchestration System for Heterogeneous Architectures},
    year = {2022},
    booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
    series = {SC'22}
}
```
