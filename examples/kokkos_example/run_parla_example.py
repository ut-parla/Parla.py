import itertools
import logging

import numpy as np

import parla
from parla.tasks import *
from parla.cpu import *
from parla.cuda import *
from parla.function_decorators import *
import parla.device
from parla.ldevice import *

import kokkos.core as kokkos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
parla.cuda.logger.setLevel(logging.DEBUG)
parla.cpu.logger.setLevel(logging.DEBUG)

######
# Example code to test kokkos integration
# Simple reduction: Sum n numbers from 1 to n on all available devices
# Check with (n*(n+1)/2)
#####

def main():
	@spawn(placement=cpu(0))
	async def test_reduction():
		print("..Starting main parla task..")

		#Configure Kokkos environment
		nCPU = 1
		nGPU = 4
		kokkos.setup(nCPU, nGPU)	
		kokkos.initialize()

		n = 10000000 #How many numbers to sum
		array = np.arange(1, n+1, dtype='float64')
		
		
		ndivisions = 10 #How many partitions of the data to take. Must be > ndevices
		results = np.zeros(ndivisions) #Store result array on CPU

		#Find devices and map parititions
		mapper = LDeviceSequenceBlocked(ndivisions)

		#Partition and send chunks of array to each device
		partition = mapper.partition_tensor(array)
		
		#Spawn ndivisions tasks and wait for them to finish (no dependencies)
		async with finish():
			for j in range(ndivisions):
				device = mapper.device(j)
				@spawn(placement=device)
				def device_local_task():
					#Grab local partition of array
					data = partition[j]
					data = data.flatten()
					#Perform reduction by calling kokkos and writing back to CPU
					results[j] = float(kokkos.reduction(data))
		#All tasks have finished
		#Sum the remaining ndivision numbers on the CPU
		s = sum(results)

		print("The sum is", s)
		#Check result
		if s == n*(n+1)/2:
			print("This is correct.")

		kokkos.finalize()
		print("..Exiting main parla task..")

if __name__ == '__main__':
	main()

	
			
				
		
				

		
