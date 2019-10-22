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
