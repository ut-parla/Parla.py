from mpi4py import MPI
import numpy as np
import kokkos.gpu.core as kokkos
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

t = time.time()
kokkos.start(rank)
t = time.time() - t
print("Kokkos init time: ", t, flush=True)


n_local = 100000000
N = size * n_local

sendbuf = None
if rank == 0:
    array = np.arange(1, N+1, dtype='float64')
    sendbuf = array.reshape(size, n_local)

t = time.time()
recvbuf = np.empty(n_local, dtype='float64')
comm.Scatter(sendbuf, recvbuf, root=0)
comm.Barrier()
t = time.time() - t

#print(rank, recvbuf, flush=True)
print(rank, "Scatter time: ", t, flush=True)


t = time.time()
result = kokkos.reduction(recvbuf, rank)
t = time.time() - t
print("Reduction time: ", t, flush=True)

t = time.time()
result = comm.gather(result, root=0)
t = time.time() - t
print(rank, "Gather time: ", t, flush=True)

if rank == 0:
    t = time.time()
    s = 0.0
    for i in range(size):
        s += result[i]
    t = time.time() - t
    print("Final Result: ", s, ((N)*(N+1))/2, flush=True)
    print("Sum time: ", t, flush=True)
        
t = time.time()
kokkos.end()
t = time.time() - t
print("Kokkos finalize time: ", t, flush=True)





