"""
A naive implementation of blocked Cholesky using Numba kernels on multiple GPUs.

Command:
 $ python blocked_cholesky_multigpu.py [block side size] [# of blocks] [# of tests]

"""

import logging
import numpy as np
from numba import jit, void, float64
import math
import time

from parla import Parla, get_all_devices
from parla.array import copy, clone_here

from parla.cuda import gpu
from parla.cpu import cpu

from parla.function_decorators import specialized
from parla.tasks import *

import cupy as cp
from cupy.cuda import cublas
from cupy.cuda import device
#from cupy.linalg import _util

from scipy import linalg
import sys

logger = logging.getLogger(__name__)

loc = gpu

gpu_arrs = []

# Configure environment
block_size = int(sys.argv[1])
n = block_size*int(sys.argv[2])
num_tests = int(sys.argv[3])

@specialized
@jit(float64[:,:](float64[:,:]), nopython=True, nogil=True)
def cholesky(a):
  """
  Naive version of dpotrf. Write results into lower triangle of the input array.
  """
  if a.shape[0] != a.shape[1]:
    raise ValueError("A square array is required.")
  for j in range(a.shape[0]):
    a[j,j] = math.sqrt(a[j,j] - (a[j,:j] * a[j,:j]).sum())
    for i in range(j+1, a.shape[0]):
      a[i,j] -= (a[i,:j] * a[j,:j]).sum()
      a[i,j] /= a[j,j]
  return a

@cholesky.variant(gpu)
def choleksy_gpu(a):
  a = cp.linalg.cholesky(a)
  return a

@specialized
@jit(float64[:,:](float64[:,:], float64[:,:]), nopython=True, nogil=True)
def ltriang_solve(a, b):
  """
  This is a naive version of dtrsm. The result is written over the input array `b`.
  """
  b = b.T
  if a.shape[0] != b.shape[0]:
    raise ValueError("Input array shapes are not compatible.")
  if a.shape[0] != a.shape[1]:
    raise ValueError("Array for back substitution is not square.")
  # For the implementation here, just assume lower triangular.
  for i in range(a.shape[0]):
    b[i] /= a[i,i]
    b[i+1:] -= a[i+1:,i:i+1] * b[i:i+1]
  return b.T

# Comments would repack the data to column - major
def cupy_trsm_wrapper(a, b):
  cublas_handle = device.get_cublas_handle()
  trsm = cublas.dtrsm
  uplo = cublas.CUBLAS_FILL_MODE_LOWER

  a = cp.array(a, dtype=np.float64, order='F', copy=False)
  b = cp.array(b, dtype=np.float64, order='F', copy=False)
  trans = cublas.CUBLAS_OP_T
  side = cublas.CUBLAS_SIDE_RIGHT

  diag = cublas.CUBLAS_DIAG_NON_UNIT
  m, n = (b.side, 1) if b.ndim == 1 else b.shape
  one = cp.array(1, dtype='d')
  trsm(cublas_handle, side, uplo, trans, diag, m, n, one, a.data.ptr, m, b.data.ptr, m)
  return b

@ltriang_solve.variant(gpu)
def ltriang_solve_gpu(a, b):
  b = cupy_trsm_wrapper(a, b)
  return b

def update_kernel(a, b, c):
  c -= a @ b.T
  return c

@specialized
def update(a, b, c):
  c = update_kernel(a, b, c)
  return c

@update.variant(gpu)
def update_gpu(a, b, c):
  c = update_kernel(a, b, c)
  return c

def cholesky_blocked_inplace(shape, num_gpus):
  """
  This is a less naive version of dpotrf with one level of blocking.
  Blocks are currently assumed to evenly divide the axes lengths.
  The input array 4 dimensional. The first and second index select
  the block (row first, then column). The third and fourth index
  select the entry within the given block.
  """
  if shape[0] * shape[2] != shape[1] * shape[3]:
    raise ValueError("A square matrix is required.")
  if shape[0] != shape[1]:
    raise ValueError("Non-square blocks are not supported.")

  # Define task spaces
  gemm1 = TaskSpace("gemm1")        # Inter-block GEMM
  subcholesky = TaskSpace("subcholesky")  # Cholesky on block
  gemm2 = TaskSpace("gemm2")        # Inter-block GEMM
  solve = TaskSpace("solve")        # Triangular solve

  for j in range(shape[0]):
    for k in range(j):
      # Inter - block GEMM
      @spawn(gemm1[j, k], [solve[j, k]], placement=[gpu(j%num_gpus)])
      def t1():
        out = get_gpu_memory(j, j, num_gpus)
        rhs = get_gpu_memory(j, k, num_gpus)
        out = update(rhs, rhs, out)
        set_gpu_memory_from_gpu(j, j, num_gpus, out)

    # Cholesky on block
    @spawn(subcholesky[j], [gemm1[j, 0:j]], placement=[gpu(j%num_gpus)])
    def t2():
      dblock = get_gpu_memory(j, j, num_gpus) 
      dblock = cholesky(dblock)
      set_gpu_memory_from_gpu(j, j, num_gpus, dblock)

    for i in range(j+1, shape[0]):
      for k in range(j):
        # Inter - block GEMM
        @spawn(gemm2[i, j, k], [solve[j, k], solve[i, k]], placement=[gpu(i%num_gpus)])
        def t3():
          out = get_gpu_memory(i, j, num_gpus)
          rhs1 = get_gpu_memory(i, k, num_gpus)
          rhs2 = get_gpu_memory(j, k, num_gpus)
          out = update(rhs1, rhs2, out)
          set_gpu_memory_from_gpu(i, j, num_gpus, out)

      # Triangular solve
      @spawn(solve[i, j], [gemm2[i, j, 0:j], subcholesky[j]], placement=[gpu(i%num_gpus)])
      def t4():
        factor = get_gpu_memory(j, j, num_gpus)
        panel  = get_gpu_memory(i, j, num_gpus)
        out = ltriang_solve(factor, panel)
        set_gpu_memory_from_gpu(i, j, num_gpus, out)

  return subcholesky[shape[0]-1]

def allocate_gpu_memory(i:int, r:int, n:int, b:int):
  with cp.cuda.Device(i):
    logger.debug("\tAllocate device:", i, "...")
    prealloced = cp.ndarray([r, n // b, b, b])
    gpu_arrs.append(prealloced)

def initialize_gpu_memory(num_gpus:int):
  if len(gpu_arrs) == num_gpus:
    for i in range(num_gpus):
      with cp.cuda.Device(i):
        gpu_arrs[i] = []


def get_gpu_memory(i:int, j:int, num_gpus:int):
  dev_id   = i % num_gpus
  local_id = i // num_gpus
  src = gpu_arrs[dev_id][local_id][j]
  dst = clone_here(src)
  return dst

def set_gpu_memory_from_gpu(i:int, j:int, num_gpus:int, v):
  dev_id   = i % num_gpus
  local_id = i // num_gpus
  gpu_arrs[dev_id][local_id][j] = v

def set_gpu_memory_from_cpu(a, num_gpus):
  for j in range(a.shape[0]):
    dev_id   = j % num_gpus 
    local_id = j // num_gpus 
    with cp.cuda.Device(dev_id):
      gpu_arrs[dev_id][local_id] = cp.asarray(a[j])

def main():
  num_gpus = cp.cuda.runtime.getDeviceCount()
  @spawn(placement=cpu)
  async def test_blocked_cholesky():
    logger.debug("Block size=", block_size, " and total array size=", n)
    assert not n % block_size

    logger.debug("Random number generate..")
    np.random.seed(10)
    # Construct input data
    a = np.random.rand(n, n)
    a = a @ a.T
    logger.debug("Random number generate done..")
    logger.debug("Copy a to a1..")
    # Copy and layout input
    a1 = a.copy()
    logger.debug("Copying done..")
    logger.debug("Shaping starts..")
    a1 = a1.reshape(n // block_size, block_size, n // block_size, block_size).swapaxes(1,2)
    logger.debug("Shaping done..")

    logger.debug("Allocate memory..")
    for i in range(num_tests):
      global gpu_arrs
      gpu_arrs = []
      for d in range(num_gpus):
        row_size = n // (block_size * num_gpus)
        if d < ((n / block_size) % num_gpus):
          row_size += 1
        if row_size > 0:
          allocate_gpu_memory(d, row_size, n, block_size)
      logger.debug("Allocate memory done..")

      set_gpu_memory_from_cpu(a1, num_gpus)

      for i in range(len(gpu_arrs)):
        logger.debug("Device ", i, " arrays are on ", gpu_arrs[i].device);

      logger.debug("Calculate starts..")
      start = time.perf_counter()

      # Call Parla Cholesky result and wait for completion
      await cholesky_blocked_inplace(a1.shape, num_gpus)

      end = time.perf_counter()
      print(end - start, "seconds")
      logger.debug("Calculate done..")

      for i in range(len(gpu_arrs)):
        logger.debug("Device ", i, " arrays are on ", gpu_arrs[i].device);

      for d in range(len(gpu_arrs)):
        logger.debug("Device:", d, " swap array..")
        gpu_arrs[d] = cp.swapaxes(gpu_arrs[d], 2, 1)
        logger.debug("Device:", d, " swap done..")

      cpu_arrs = np.empty([n // block_size,
                           block_size,
                           n // block_size,
                           block_size], dtype=float)
      for r_num in range(n // block_size):
        dev_id   = r_num % num_gpus
        local_id = r_num // num_gpus
        cpu_arrs[r_num] = cp.asnumpy(gpu_arrs[dev_id][local_id])
      cpu_arrs = cpu_arrs.reshape(n, n)
      print("Truth", linalg.cholesky(a).T)

      # Check result
      computed_L = np.tril(cpu_arrs)
      print("Soln", computed_L)
      error = np.max(np.absolute(a-computed_L @ computed_L.T))
      print("Error", error)
      assert(error < 1E-8)
      del gpu_arrs

if __name__ == '__main__':
    with Parla():
        main()
