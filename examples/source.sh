export PARLA_ROOT=$(pwd)
export MAGMA_ROOT=$PARLA_ROOT/artifact/magma
export CUBLASMG_ROOT=$PARLA_ROOT/artifact/cublasmg/cublasmg
export CUDAMG_ROOT=$PARLA_ROOT/artifact/cublasmg/cudalibmg

#Please set your CUDA_ROOT
#export CUDA_ROOT=<your cuda installation here>
#export CUDA_ROOT=<your parla_env path here>
export CUDA_ROOT=$CONDA_PREFIX

#Optional: Set PYTHONPATH
#export PYTHONPATH = $PARLA_ROOT:$PYTHONPATH

