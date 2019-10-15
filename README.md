# Parla

Parla is a high-level programming system for running numerical simulations on heterogeneous architectures.
The current prototype emphasizes orchestrating data movement and kernel calls across all the CPUs and GPUs available on a given machine.

# Installation

Parla is available as a conda package. A docker image with the conda package already set up is also available. Parla requires Python 3.7, numpy, cupy, scipy, and numba (currently needed only for the examples).

## Installation with Conda

To use the conda package, you must first install miniconda.
To install miniconda you can follow the detailed instructions available at [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
Abbreviated instructions are included here.
If you are running Linux and have `wget` available, you can download and install miniconda into the miniconda subdirectory of your home directory by running

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
```

To make miniconda available on your path in a given terminal session run
```
export PATH=$HOME/miniconda/bin:$PATH
source activate
```

Once that's done, you can install parla by running

```
conda install -y -c insertinterestingnamehere parla
```

If you have already installed parla but need to access your miniconda installation from a new terminal session just run (as before)
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

If git is not available, you can install it as a conda package alongside parla by running `conda install -y git` from a terminal session configured to use miniconda.

## Running the Docker Container

To get a shell inside the provided docker container run

```
docker run --gpus all --rm -it insertinterestingnamehere/parla
```

Depending on how your machine is set up, you may need to run this command as root using `sudo` or some other method.
Since cuda is required for all the demos, you must provide some GPUs for the docker container to use.
For this to work using the command shown, you need to use Docker 19.03 or later.
