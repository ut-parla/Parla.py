# Parla

Parla is a high-level programming system for running numerical simulations on heterogeneous architectures.
The current prototype emphasizes orchestrating data movement and kernel calls across all the CPUs and GPUs available on a given machine.
<!--API documentation is available at [http://www.cs.utexas.edu/~amp/psaap/Parla.py/index.html](http://www.cs.utexas.edu/~amp/psaap/Parla.py/index.html). -->


# Installation

Parla is currently distributed from this repository as a Python module.
In the future, Parla will be available as a Conda package; for now, it must manually be installed.
For new users unfamiliar with Python package management, we recommend using Miniconda to manage Parla and its dependencies.
To install Miniconda you can follow the detailed instructions available from [Miniconda's documentation](https://docs.conda.io/en/latest/miniconda.html).
Abbreviated instructions are included here.
If you are running Linux and have `wget` available, you can download and install Miniconda into the Miniconda subdirectory of your home directory by running

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
rm miniconda.sh
```

Restart your shell for changes to take effect.

<!--Parla is available as a Conda package. -->
<!--A docker image with the Conda package already set up is also available. -->
Parla requires Python>=3.7, numpy, cupy, cython, and psutil.
Features for VECs (dlm_open execution spaces) also depends on the C package libunwind.

To run all examples you require: scipy, numba, pexpect, mkl, mkl-service, and cupy.
The synthetic submodule requires compilation of its "busy sleep" routines for GPU and CPUs. 
See the examples/synthetic/README for details. 

Note that mkl-service is REQUIRED to find and import the mkl module. 
They can be installed with conda: `conda install -c conda-forge mkl mkl-service`
They are used to control the number of threads used by linear algebra routines to prevent oversubscription. 

You may want to create a new Conda environment with the required Python version, like so

```
conda create -n environment_name python=3.7
```

If you have sudo privileges on your system, install libunwind-dev as follows:

```
sudo apt-get install libunwind-dev # Installs libunwind on your system
```

If you do not have sudo privileges and libunwind-dev is not already installed, you will have to build it yourself.
The repository and build instructions are located [here](https://github.com/libunwind/libunwind).

To activate your Conda environment and install the other required dependencies, run

```
conda activate environment_name # Opens your Conda environment
conda install numpy cython psutil scipy numba cupy pexpect # Installs Python packages into your environment
```

To install Parla itself, navigate to the top-level directory of this repository, and from it, run ONE of the following two commands:

```
pip install .     # For Parla Users
pip install -e .  # For Parla Developers who are modifying Parla and would like to see their changes reflected as they work
```

The installation process creates extra files in the repository.
Virtual execution contexts (experimental - see below) require on some of these files to be present.
If you are not using virtual execution contexts and would like to clear out the extra files created by Parla on installation, use [`git clean`](https://git-scm.com/docs/git-clean).

Now all the scripts in this repository are runnable as normal Python scripts.
To test your installation, try running

```
python tutorial/0_hello_world/hello.py
```

This should print

```
Hello, World!
```

We recommend entering the tutorial directory and working through it as a starting point for learning Parla.

To run examples for blr, nbody, and synthetic graphs you have to initalize the submodules.
We recommend running:
```
git submodule update --init --recursive --remote
```

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

## Acknowledgements
This software is based upon work supported by the Department of Energy, National Nuclear Security Administration under Award Number DE-NA0003969.

## How to Cite Parla.py
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6941889.svg)](https://doi.org/10.5281/zenodo.6941889)
