import numpy as np
import cupy as cp
from time import time
with cp.cuda.Device(0):
    A = cp.random.rand(1000,100, dtype=cp.float32)
    Q,R=cp.linalg.qr(A,mode='reduced')
with cp.cuda.Device(1):
    A = cp.random.rand(1000,100, dtype=cp.float32)
    Q,R=cp.linalg.qr(A,mode='reduced')
M=500000
N=1000
if 1:
    print( f'Single GPU code, QR {M}-by-{N}')
    Q0 = cp.random.randn(M,N,dtype=cp.float32)
    tic =time()
    Q0,R0=cp.linalg.qr(Q0,mode='reduced')
    toc = time()
    print(f'QR time {toc-tic}')
if 0:
    print( f'Two GPU, QR {M}-by-{N} per GPU')
    tic = time()
    with cp.cuda.Device(0):
        Q0 = cp.random.randn(M,N,dtype=cp.float32)
        Q0,R0=cp.linalg.qr(Q0,mode='reduced')
    with cp.cuda.Device(1):
        Q1 = cp.random.randn(M,N,dtype=cp.float32)
        Q1,R1=cp.linalg.qr(Q1,mode='reduced')
        R1=R1.flatten()
    with cp.cuda.Device(0):
        R01 = cp.asarray(R1)
        Q = cp.zeros((2*N,N),dtype=cp.float32)
        Q[:N,:N]=R0
        Q[N:2*N,:N]= cp.reshape(R01,(N,N))
        Q,R= cp.linalg.qr(Q,mode='reduced')
        Q01=Q[N:2*N,:N]
        Q01=Q01.flatten()
    with cp.cuda.Device(1):
        Q10 = cp.asarray(Q01)
        Q10 = cp.reshape(Q10,(N,N))
        Q1 = Q1@Q10
    with cp.cuda.Device(0):
        Q0 = Q0@Q[:N,:N]
    toc = time()
    print(f'QR time {toc-tic}')
    #print('Output should be Q0,Q1, R')
