import os
import sys

#Figure 10: Cholesky
def run_cholesky_28():
    pass

#Figure 13: Parla Cholesky (CPU)
def run_cholesky_20_host():
    pass

#Figure 13: Parla Cholesky (GPU)
def run_cholesky_20_gpu():
    pass

#Figure 13: Dask Cholesky (CPU)
def run_dask_cholesky_20_host():
    pass

#Figure 13: Dask Cholesky (GPU)
def run_dask_cholesky_20_gpu():
    pass

#Figure 10: Jacobi
def run_jacobi():
    pass


#Figure 10: Matmul
def run_matmul():
    pass

#Figure 10: BLR
def run_blr():
    pass

#Figure 10: NBody
def run_nbody():
    pass

#Figure 10: Synthetic Reduction
def run_reduction():
    pass

#Figure 10: Synthetic Independent
def run_independent():
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

figure_10 = [run_cholesky_28, run_jacobi, run_matmul, run_blr, run_nbody, run_reduction, run_independent, run_serial]
figure_13 = [run_cholesky_20_host, run_cholesky_20_gpu, run_dask_cholesky_20_host, run_dask_cholesky_20_gpu]
figure_15 = [run_batched_cholesky]
figure_12 = [run_prefetching_test]



