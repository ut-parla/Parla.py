from core cimport *
import numpy as np
cimport numpy as np

def start(dev=0):
	init(dev)

def end():
	finalize()

def reduction(array):
	cdef int N = len(array)
	cdef long device = -1
	cdef double result
	cdef double[:] arr
	
	cdef long temp
	cdef double* parr	
	if isinstance(array, (np.ndarray, np.generic)):
		arr = array
		result = kokkos_function(&arr[0], N, device)
	else: #assume CuPy array 
		temp = <long> array.data.mem.ptr
		device = <long> array.data.device.id
		parr = <double *> temp
		result = kokkos_function(parr, N, device)
	return result


