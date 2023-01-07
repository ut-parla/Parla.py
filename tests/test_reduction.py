"""
This script tests the evaluation of a multi-device reduction using
PArrays. The reduction involves a multi-dimensional array and sums
across the device memory, storing the final result in a PArray.
"""

import argparse
import os
from os.path import expanduser, join
import sys

# Parse the command line data
parser = argparse.ArgumentParser(description='Multi-GPU reduction using PArrays')
parser.add_argument('-ngpus', type=int,
                    default=2, help="Number of gpus to use.")
parser.add_argument('-blocks', type=int, 
                    default=2, help="Number of element blocks/partitions.")
parser.add_argument('-trials', type=int,
                    default=2, help="Number of repititions for timing regions of code.")
parser.add_argument('-N', type=int,
                    default=10, help="Number of elements/block.")

args = parser.parse_args()

# Display the arguments in the console
print("\nOptions used:\n")

for arg in vars(args):
    print(arg, "=", getattr(args, arg))

print("\n")

ngpus = args.ngpus
blocks = args.blocks
trials = args.trials
N = args.N

# Error checking for the input
assert blocks >= ngpus, "Error: blocks >= ngpus.\n"

# Specify information about the devices
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    print("CUDA_VISIBLE_DEVICES is not set. Assuming 0-3")
    cuda_visible_devices = list(range(4))
else:
    cuda_visible_devices = cuda_visible_devices.strip().split(',')
    cuda_visible_devices = list(map(int, cuda_visible_devices))

gpus = cuda_visible_devices[:ngpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

dirname = os.path.dirname(__file__)
parla_dir = dirname + "/../Parla.py/"
sys.path.append(parla_dir)

import numpy as np
import cupy as cp
import numba as nb
from numba import cuda
import time

# Parla modules and decorators
# These should be imported after "CUDA_VISIBLE_DEVICES" is set
from parla import Parla, get_all_devices, parray
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import spawn, TaskSpace

def main():

    @spawn(placement=cpu)
    async def main_task():

        # Create a 2-D array on the host then
        # wrap it with a PArray
        A = np.zeros([blocks,N], dtype=np.int64)
        A_pa = parray.asarray(A)

        # Create the PArray that will hold the reduction
        reduction_result_pa = parray.asarray( np.zeros([N]) )
        print(f"1 reduction_result_pa: {reduction_result_pa}")

        # Each row of the array A will be the row_id
        # When we sum down the rows for the reduction
        # the correct answer is (blocks - 1)*(blocks)/2
        solution = (blocks - 1)*(blocks)/2

        print("Solution for this configuration:", solution, "\n")

        # Create the task space for Parla
        sts = TaskSpace("SetupTaskSpace")
        rts = TaskSpace("ReductionTaskSpace")

        # Store the times for statistics
        parla_times = np.zeros([trials]) 

        for trial_idx in range(trials):

            # Setup the array
            for i in range(blocks):
                print(f"{i}")
                # Map the block to a specific device
                dev_idx = i % ngpus
                
                # Set the values in the PArray to be the block idx
                @spawn(taskid=sts[trial_idx,i], placement=gpu[dev_idx], output=[A_pa[i]])
                def init_task():
                    print(f"initialize i: {i}, dev_id: {dev_idx}")
                    A_pa[i][:] = i

                await sts
    
            # Now print the value of the PArray to the console for inspection
            print("Initialization is complete. Printing the output now...\n")
            print("A_pa =", A_pa, "\n")
            print("Preparing to start the reduction task.\n")

            parla_start = time.perf_counter()

            # Perform the reduction of the PArray on the device
            @spawn(taskid=rts[trial_idx], placement=gpu[0], input=[A_pa], 
                   output=[reduction_result_pa], dependencies=[sts[trial_idx,0:blocks]])
            def reduction_task():

                # We can use the update() method here since the result array is write-only
                print(f"cp.sum(A_pa.array, axis=0): {cp.sum(A_pa.array, axis=0)}")
                print(f"2 reduction_result_pa: {reduction_result_pa}")
                reduction_result_pa[:] = cp.sum(A_pa.array, axis=0)
                print(f"3 reduction_result_pa: {reduction_result_pa}")
                print(f"A_pa: {A_pa}")

            await rts

            parla_end = time.perf_counter()
            parla_times[trial_idx] = parla_end - parla_start

        # End of the trial loop

        dts = TaskSpace("DataTaskSpace")
        @spawn(taskid=dts[0], placement=cpu, input=[reduction_result_pa])
        def gather_data():
            pass
        
        await dts

        print("type( reduction_result_pa.array ) =", type( reduction_result_pa.array ) )
        print("reduction_result_pa.array  =", reduction_result_pa.array )

        print("Parla times (min, max, avg) [s] ", parla_times.min(),",",
        parla_times.max(),",", parla_times.mean(), "\n",flush=True)

        # Check for correctness
        print("Is the solution correct?", np.all(reduction_result_pa.array == solution), "\n")


if __name__ == "__main__":

    # First create the Parla context before spawning tasks
    with Parla():

        main()