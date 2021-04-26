# Parla

Parla is a high-level programming system for running numerical simulations on heterogeneous architectures.
The current prototype emphasizes orchestrating data movement and kernel calls across all the CPUs and GPUs available on a given machine.
API documentation is available at [http://www.cs.utexas.edu/~amp/psaap/Parla.py/index.html](http://www.cs.utexas.edu/~amp/psaap/Parla.py/index.html).

# Installation

Parla is available as a Conda package. 
A docker image with the Conda package already set up is also available. 
Parla requires Python 3.7 and numpy. The examples also require scipy, numba, and cupy.

## Installation with Conda

To use the conda package, you must first install Miniconda.
To install Miniconda you can follow the detailed instructions available at [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
Abbreviated instructions are included here.
If you are running Linux and have `wget` available, you can download and install Miniconda into the Miniconda subdirectory of your home directory by running

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
```

To make Miniconda available on your path in a given terminal session run
```
export PATH=$HOME/miniconda/bin:$PATH
source activate
```

Once that's done, you can install parla by running

```
conda install -y -c ut-parla parla
```

If you have already installed parla but need to access your Miniconda installation from a new terminal session just run (as before)
```
export PATH=$HOME/miniconda/bin:$PATH
source activate
```

Once parla is installed and your environment is configured to use it, all the scripts in this repository's examples directory are runnable as normal python scripts.
If git is installed you can clone the repository and run the inner product example by running:

```
git clone https://github.com/ut-parla/Parla.py
python Parla.py/examples/inner.py
```

If git is not available, you can install it as a Conda package alongside parla by running `conda install -y git` from a terminal session configured to use Miniconda.

## Running the Docker Container

The Parla container requires CUDA support in the Docker host environment.
To get a shell inside the provided docker container run

```
docker run --gpus all --rm -it utparla/parla
```

Depending on your Docker configuration, you may need to run this command as root using `sudo` or some other method.
Since CUDA is required for all the demos, you must provide some GPUs for the docker container to use.
For this to work using the command shown, you need to use Docker 19.03 or later.

## Virtual Execution Contexts (Experimental)

VECs are currently experimental.
Some of the packaging work for them still needs to be done.
Here are instructions for how to get the VEC prototype running locally:

Glibc is usually the pain point.
Everything should be working for numpy/openblas now.
Other libraries may or may not work.
Make sure you have recent versions of gcc, binutils, make, and CMake.
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
python setup.py install;
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
