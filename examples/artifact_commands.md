# Cholesky 28k

## Generate Matrix
python make_cholesy_input.py -n 28000

## Automatic Movement, Policy Placement
python blocked_cholesky_automatic.py -b 2000 -nblocks 14 -trials 1 -matrix chol_28000.npy -fixed 0

## Automatic Movement, User Placement
python blocked_cholesky_automatic.py -b 2000 -nblocks 14 -trials 1 -matrix chol_28000.npy -fixed 1

## Manual Movement, User Placement
python blocked_cholesky_manual.py -b 2000 -nblocks 14 -trials 1 -matrix chol_28000.npy -fixed 1

# Matmul (Add correctness check?)

## Automatic Movement, Policy Placement
python matmul_automatic.py -n 32000 -trials 1 -fixed 0

## Automatic Movement, User Placement
python matmul_automatic.py -n 32000 -trials 1 -fixed 1

## Manual Movement, User Placement
python matmul_manual.py -n 32000 -trials 1 -fixed 1

# Jacobi

## Automatic Movement, Policy Placement
python jacobi_automatic.py -trials 1 -fixed 0

## Automatic Movement, User Placement
python jacobi_automatic.py -trials 1 -fixed 1

## Manual Movement, User Placement
python jacobi_manual.py -trials 1 -fixed 1


# BLR


# NBody

## Automatic Movement, Policy Placement
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla[NUM GPUs]_eager_sched.ini

## Automatic Movement, User Placement
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla[NUM GPUs]_eager.ini

## Manual Movement, User Placement
python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla[NUM GPUs].ini
