# Parla

Parla is a prototype high level system for orchestrating data movement and kernel calls for all the CPUs and GPUs available on a given machine.

# Installation

Parla is available as a conda package. A docker image with the conda package already set up is also available. Parla requires Python 3.7, numpy, cupy, scipy, and numba (currently needed only for the examples)

To use the conda package, you must first install miniconda.
To install miniconda you can follow the detailed instructions available at [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
Abbreviated instructions are included here.
If you are running Linux and have `wget` available, you can download and install miniconda into the miniconda subdirectory of your home directory by running

```
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh
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
conda install -c insertinterestingnamehere parla
```

If you have already installed parla but need to access your miniconda installation from a new terminal session just run (as before)
```
export PATH=$HOME/miniconda/bin:$PATH
source activate
```

A docker container with the above steps already completed is available at [https://hub.docker.com/r/insertinterestingnamehere/parla](https://hub.docker.com/r/insertinterestingnamehere/parla). You can get a shell in this docker container by running

```
docker run -it insertinterestingnamehere/parla
```

Once parla is installed and your environment is configured to use it, all the scripts in this repository's examples directory are runnable as normal python scripts.
If git is installed you can clone the repository and run the inner product example by running:

```
git clone https://github.com/ut-parla/Parla.py
python Parla.py/examples/inner.py
```

If git is not available, you can install it as a conda package alongside parla by running `conda install git` from a terminal session configured to use miniconda.

