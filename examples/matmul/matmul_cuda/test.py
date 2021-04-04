import cupy as cp
import mult.core
import numpy as np
N = 10
M = 5
with cp.cuda.Device(1):
    A = cp.asarray(cp.random.rand(N, M), dtype=np.float32, order='F')
    B = cp.asarray(cp.ones((M, N)), dtype=np.float32, order='F')
    C = cp.asarray(cp.zeros((N, N)), dtype=np.float32, order='F')
    print(A.shape, B.shape, C.shape)

    mult.core.gemm(A, B, C, 1)


    print(C)

    C2 = A @ B
    print(C2.T)
