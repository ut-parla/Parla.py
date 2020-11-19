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

itemsize = MPI.DOUBLE.Get_size()
if rank == 0:
    nbytes = N*itemsize
else:
    nbytes = 0

t = time.time()
#Create shared block
win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
#Create numpy array in shared memory
buf, itemsize = win.Shared_query(0)
assert itemsize == MPI.DOUBLE.Get_size()
array = np.ndarray(buffer=buf, dtype='d', shape=(N,))
t = time.time() -t

print(rank, "Allocate shared block time: ", t, flush=True)

if rank == 0:
    array[:N] = np.arange(1, N+1, dtype='float64')
comm.Barrier()

start = (rank)*n_local
end = (rank+1)*n_local

t = time.time()
result = kokkos.reduction(array[start:end], rank)
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





