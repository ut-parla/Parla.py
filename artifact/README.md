#Repositories for 3rd Party Performance Comparisons

## Cublasmg

For SC22 Artifact review, we provide Cublasmg here, which is also available from: https://developer.nvidia.com/cudamathlibraryea

First, you should set environment variables. We provide a source file to do this.
The source file sets `CUBLASMG_ROOT` and `CUDAMG_ROOT` paths.

```
$ export PARLA_ROOT=[Your Parla root directory path]
$ cd $PARLA_ROOT
$ source examples/source.sh
```

By default this sets your CUDA_ROOT to be the location of your current conda env. 
If you have cudatoolkit=10.1 and cudatoolkit-dev=10.1 installed (such as through the provided requirements.txt) then this should be sufficient. 

Otherwise, set your CUDA_ROOT to the your CUDA installation.
Note: This requires CUDA/10.1 and gcc>=8.3.0. 


In `CUBLASMG_ROOT/test`, we have the modified block matrix multiplication file
to perform: C = A @ B.T at the same size as Parla.

You must compile the examples in the test folder.
(Note that it requires CUDA and the set paths above)

```
$ cd $CUBLASMG_ROOT/test
$ make
```

This should make three tests that can be run as the following:

```
#Run 32k x 32k Matrix Mult on 1 GPU
$ ./1gpuGEMM.exe

#Run 32k x 32k Matrix Mult on 2 GPUs
$ ./2gpuGEMM.exe

#Run 32k x 32 Matrix Mult on 4 GPUs
$ ./4gpuGEMM.exe
```

## Magma

For convience we include the Magma linear algebra library as a submodule.
Unless you have a local installation, set the Magma's root path.

```
$ export MAGMA_ROOT=PARLA_ROOT/artifact/magma
```

Instructions for compiling Magma are included in the `$MAGMA_ROOT/README`.
To run the cholesky comparison you must build Magma with testing enabled. 
This is the default in Magma 2.6.

To compare we use MAGMA_ROOT/testing/testing_dpotrf_mgpu executable.

The tests can be performed as:

```
$ ./testing_dpotrf_mgpu -N 28000 --ngpu 1
$ ./testing_dpotrf_mgpu -N 28000 --ngpu 2
$ ./testing_dpotrf_mgpu -N 28000 --ngpu 4
```


## Dask

To enable users to reproduce Cholesky comparison experiments, we include the conda
environment file that we used.

Before running the Dask Cholesky evaluations, please run the following commands:

```
# Create a conda environment from the provided environment file.
$ conda create --name <env> --file "$PARLA_ROOT/dask-requirements.txt"
```

More detailed information of the installations can be found from the official
Dask web pages:

Dask: https://www.dask.org

Dask-distributed: https://distributed.dask.org/en/stable

Dask-cuda: https://docs.rapids.ai
