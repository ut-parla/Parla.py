# Note
All commands should be execute in the top level directory of this repo

# Cholesky 28k

## Generate Matrix
python examples/cholesky/make_cholesky_input.py -n 28000

## Automatic Movement, Policy Placement
python examples/cholesky/blocked_cholesky_automatic.py -b 2000 -nblocks 14 -trials 1 -matrix chol_28000.npy -fixed 0

## Automatic Movement, User Placement
python examples/cholesky/blocked_cholesky_automatic.py -b 2000 -nblocks 14 -trials 1 -matrix chol_28000.npy -fixed 1

## Manual Movement, User Placement
python examples/cholesky/blocked_cholesky_manual.py -b 2000 -nblocks 14 -trials 1 -matrix chol_28000.npy -fixed 1

# Matmul (Add correctness check?)

## Automatic Movement, Policy Placement
python examples/matmul/matmul_automatic.py -n 32000 -trials 1 -fixed 0

## Automatic Movement, User Placement
python examples/matmul/matmul_automatic.py -n 32000 -trials 1 -fixed 1

## Manual Movement, User Placement
python examples/matmul/matmul_manual.py -n 32000 -trials 1 -fixed 1

# Jacobi

## Automatic Movement, Policy Placement
python examples/jacobi/jacobi_automatic.py -trials 1 -fixed 0

## Automatic Movement, User Placement
pythonexamples/jacobi/jacobi_automatic.py -trials 1 -fixed 1

## Manual Movement, User Placement
python examples/jacobi/jacobi_manual.py -trials 1 -fixed 1


# BLR

## Input File Generation

mkdir examples/blr/inputs
python examples/blr/app/main.py -mode gen -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k -b 2500 -nblocks 4

## Parla
python examples/blr/app/main.py -mode run -type mgpu_blr -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k.npy -b 2500 -nblocks 4 -ngpus {n_gpus}
## Python Threading

python examples/blr/app/main.py -mode run -type parla -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k.npy -b 2500 -nblocks 4 -fixed {0|1} -movement {eager|lazy} -ngpus {n_gpus}

# NBody

## Input File Generation

mkdir examples/nbody/python-bh/input
python examples/nbody/python-bh/bin/gen_input.py normal 10000000 examples/nbody/python-bh/input/n10M.txt

## Parla

### Automatic Movement, Policy Placement
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla[NUM GPUs]_eager_sched.ini

### Automatic Movement, User Placement
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla[NUM GPUs]_eager.ini

### Manual Movement, User Placement
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla[NUM GPUs].ini

## Python Threading Implementation

### 1 GPU
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/singlegpu.ini

### 2 GPU
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/2gpus.ini

### 3 GPU
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/4gpus.ini
