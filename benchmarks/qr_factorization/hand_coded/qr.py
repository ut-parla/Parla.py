import argparse
import numpy as np
import cupy as cp
from time import time

def main():
    # Warmup I guess?
    with cp.cuda.Device(0):
        # Default data type for cp.random.rand is double: https://docs.cupy.dev/en/stable/reference/generated/cupy.random.rand.html
        A = cp.random.rand(1000, 100, dtype=PRECISION)
        Q,R=cp.linalg.qr(A,mode='reduced') # 'reduced' is the default mode, so mine is the same
    
    with cp.cuda.Device(1):
        A = cp.random.rand(1000, 100, dtype=PRECISION)
        Q,R=cp.linalg.qr(A,mode='reduced')
    
    # Single GPU
    if NGPUS == 1:
        print( f'Single GPU code, QR {M}-by-{N}, {args.precision} precision')
        Q0 = cp.random.rand(M, N, dtype=PRECISION)
        tic =time()
        cp.cuda.stream.get_current_stream().synchronize() # Added sync to time only the kernel
        Q0,R0=cp.linalg.qr(Q0,mode='reduced')
        cp.cuda.stream.get_current_stream().synchronize() # Added sync to time only the kernel
        toc = time()
        print(f'QR time {toc-tic}')
    
    # Multi-GPU
    if NGPUS == 2:
        print( f'Two GPU, QR {M}-by-{N} per GPU, {args.precision} precision, tolerance = {TOLERANCE}')
    
        # Set up the arrays on the GPUs before starting the timer.
        # Also have a CPU-side array for checking later (this won't affect timing)
        with cp.cuda.Device(0):
            Q0 = cp.random.randn(M, N, dtype=PRECISION)
            cpu_Q0_orig = cp.asnumpy(Q0)
            cp.cuda.stream.get_current_stream().synchronize()
            Q = cp.zeros((2*N,N), dtype=PRECISION) # Moved this initialization to outside the timed section
        with cp.cuda.Device(1):
            Q1 = cp.random.randn(M, N, dtype=PRECISION)
            cpu_Q1_orig = cp.asnumpy(Q1)
            cp.cuda.stream.get_current_stream().synchronize()
    
        # Timer starts after setup is complete
        # Same algorithm as George's
        tic = time()
        with cp.cuda.Device(0):
            Q0,R0=cp.linalg.qr(Q0,mode='reduced')
        with cp.cuda.Device(1):
            Q1,R1=cp.linalg.qr(Q1,mode='reduced')
            R1=R1.flatten()
        with cp.cuda.Device(0):
            R01 = cp.asarray(R1)
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
    
        # Sync before calling the end timer
        with cp.cuda.Device(0):
            cp.cuda.stream.get_current_stream().synchronize()
        with cp.cuda.Device(1):
            cp.cuda.stream.get_current_stream().synchronize()
        toc = time()
    
        print(f'QR time {toc-tic}')
    
        ## Check product
        A = np.vstack((cpu_Q0_orig, cpu_Q1_orig))
        cpu_Q0 = cp.asnumpy(Q0)
        cpu_Q1 = cp.asnumpy(Q1)
        cpu_Q = np.vstack((cpu_Q0, cpu_Q1))
        cpu_R = cp.asnumpy(R)
    
        # Check product
        prod = np.matmul(cpu_Q, cpu_R, dtype=PRECISION)
        is_correct_prod = np.allclose(prod, A, atol=TOLERANCE)
        
        # Check for orthonormal Q
        Q_check = np.matmul(cpu_Q.transpose(), cpu_Q)
        is_ortho_Q = np.allclose(Q_check, np.identity(N, dtype=PRECISION), atol=TOLERANCE)
        
        # Check for upper triangular R
        is_upper_R = np.allclose(cpu_R, np.triu(cpu_R), atol=TOLERANCE)
    
        print(is_correct_prod and is_ortho_Q and is_upper_R)

# Added some argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rows", help="Number of rows for input matrix; must be >> cols", type=int, default=5000)
    parser.add_argument("-c", "--cols", help="Number of columns for input matrix", type=int, default=100)
    parser.add_argument("-g", "--ngpus", help="Sets number of GPUs to run on. If set to more than you have, undefined behavior", type=int, default='2')
    parser.add_argument("-p", "--precision", help="Sets precision", default='double')
    args = parser.parse_args()

    M = args.rows
    N = args.cols
    NGPUS = args.ngpus
    if args.precision == 'double':
        PRECISION = cp.float64
        TOLERANCE = 1e-08
    elif args.precision == 'single':
        PRECISION = cp.float32
        TOLERANCE = 1e-03
    
    main()
