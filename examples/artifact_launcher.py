import os
import sys
import argparse
import pexpect as pe

#######
# Define functions to gather each result (per figure, per app)
#######

#Figure 10: Cholesky
def run_cholesky_28(gpu_list):

    for n_gpus in gpu_list:
        command = f"python test_script.py -gpus {n_gpus}"
        output = pe.run(command, timeout=100, withexitstatus=True)
        assert(output[1] == 0)


#Figure 13: Parla Cholesky (CPU)
def run_cholesky_20_host():
    pass

#Figure 13: Parla Cholesky (GPU)
def run_cholesky_20_gpu(gpu_list):
    pass

#Figure 13: Dask Cholesky (CPU)
def run_dask_cholesky_20_host(cores_list):
    pass

#Figure 13: Dask Cholesky (GPU)
def run_dask_cholesky_20_gpu(gpu_list):
    pass

#Figure 10: Jacobi
def run_jacobi(gpu_list):
    pass


#Figure 10: Matmul
def run_matmul(gpu_list):
    pass

#Figure 10: BLR
def run_blr(gpu_list):
    pass

#Figure 10: NBody
def run_nbody(gpu_list):
    pass

#Figure 10: Synthetic Reduction
def run_reduction(gpu_list):
    pass

#Figure 10: Synthetic Independent
def run_independent(gpu_list):
    pass

#Figure 10: Synthetic Serial
def run_serial():
    pass

#Figure 15: Batched Cholesky Variants
def run_batched_cholesky():
    pass

#Figure 12: Prefetching Plot
def run_prefetching_test():
    pass

#Figure 14: GIL test
def run_GIL_test():
    pass

test = [run_cholesky_20_host]
figure_10 = [run_cholesky_28, run_jacobi, run_matmul, run_blr, run_nbody, run_reduction, run_independent, run_serial]
figure_13 = [run_cholesky_20_host, run_cholesky_20_gpu, run_dask_cholesky_20_host, run_dask_cholesky_20_gpu]
figure_15 = [run_batched_cholesky]
figure_12 = [run_prefetching_test]


for f in test:
    f()





