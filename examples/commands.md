# Note
All commands should be execute in the top level directory of this repo (PARLA_ROOT)
Before running any 3rd party examples, make sure you have read the README in PARLA_ROOT/artifact and compiled the relvant tests. 
Please see source examples/source.h for an example (recommended) configuration. 

Below we list all of the apps (in artifact_launcher) and how they can be run by themselves. 

All scripts aside from `nbody` should have an updated argparse. Please run '-h' to see a list of valid options for each. 

# Cholesky 28k (run_cholesky_28k)

## Generate Matrix
python examples/cholesky/make_cholesky_input.py -n 28000

## Automatic Movement, Policy Placement
python examples/cholesky/blocked_cholesky_automatic.py -b 2000 -nblocks 14 -trials 1 -matrix examples/cholesky/chol_28000.npy -fixed 0

## Automatic Movement, User Placement
python examples/cholesky/blocked_cholesky_automatic.py -b 2000 -nblocks 14 -trials 1 -matrix examples/cholesky/chol_28000.npy -fixed 1

## Manual Movement, User Placement
python examples/cholesky/blocked_cholesky_manual.py -b 2000 -nblocks 14 -trials 1 -matrix examples/cholesky/chol_28000.npy -fixed 1

# Matmul (run_matmul)

## Automatic Movement, Policy Placement
python examples/matmul/matmul_automatic.py -n 32000 -trials 1 -fixed 0

## Automatic Movement, User Placement
python examples/matmul/matmul_automatic.py -n 32000 -trials 1 -fixed 1

## Manual Movement, User Placement
python examples/matmul/matmul_manual.py -n 32000 -trials 1 -fixed 1

# Jacobi (run_jacobi)

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

## Parla (run_blr_parla)
python examples/blr/app/main.py -mode run -type mgpu_blr -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k.npy -b 2500 -nblocks 4 -ngpus {n_gpus}
## Python Threading

python examples/blr/app/main.py -mode run -type parla -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k.npy -b 2500 -nblocks 4 -fixed {0|1} -movement {eager|lazy} -ngpus {n_gpus}

# NBody

## Input File Generation

mkdir examples/nbody/python-bh/input
python examples/nbody/python-bh/bin/gen_input.py normal 10000000 examples/nbody/python-bh/input/n10M.txt

## Parla (run_blr_threads)

### Automatic Movement, Policy Placement
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla{NUM GPUs}_eager_sched.ini

### Automatic Movement, User Placement
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla{NUM GPUs}_eager.ini

### Manual Movement, User Placement
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla{NUM GPUs}.ini

## Python Threading Implementation (run_nbody_threads)

### 1 GPU
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/singlegpu.ini

### 2 GPU
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/2gpus.ini

### 3 GPU
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/4gpus.ini


# Synthetic Independent (run_independent)

## Generate input graph

mkdir examples/synthetic/inputs
python examples/synthetic/graphs/generate_independent_graph.py -overlap 0 -width 300 -N 6250 -gil_time 0 -user 1 -weight 16000 -output examples/synthetic/inputs/independent.gph

## Run graph

python examples/synthetic/run.py -graph examples/synthetic/inputs/independent.gph -d 1000 -loop 3 -reinit 2 -data_move {1=manual|2=automatic} -user {0|1}


# Synthetic Reduction (run_reduction)

## Generate input graph

mkdir examples/synthetic/inputs
python examples/synthetic/graphs/generate_reduce_graph.py -overlap 1 -level 8 -branch 2 -N 6250 -gil_time 0 -weight 16000 -user 1 -output examples/synthetic/inputs/reduction.gph

## Run graph

python examples/synthetic/run.py -graph examples/synthetic/inputs/reduction.gph -d 1000 -loop 6 -reinit 2 -data_move {1=manual|2=automatic} -user {0|1}

# Synthetic Serial (run_serial)

## Generate input graph

mkdir examples/synthetic/inputs
python examples/synthetic/graphs/generate_serial_graph.py -level 150 -N 6250 -gil_time 0 -weight 16000 -user 1 -output examples/synthetic/inputs/serial.gph

## Run graph

python examples/synthetic/run.py -graph examples/synthetic/inputs/serial.gph -d 1000 -loop 6 -reinit 2 -data_move {1=manual|2=automatic} -user {0|1}

# Parla Scaling Tests (run_independent_parla_scaling)

python examples/synthetic/run.py -graph examples/synthetic/artifact/graphs/independent_1000.gph -threads {thread_count} -data_move 0 -weight {task_size} -use_gpu 0 -gweight {gil_time}

# Dask Scaling Tests (run_independent_dask_thread_scaling)

python examples/synthetic/artifact/scripts/run_dask_thread_gil.py -workers {thread_count} -time {task_size} -gtime {gil_time} -n 1000

# Third-Party

## MAGMA Cholesky (run_cholesky_magma)

See artifact/README or the function in examples/artifact_launcher.py

## CUBLAS Matmul

See artifact/README or the function in examples/artifact_launcher.py


