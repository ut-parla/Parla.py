import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
#Blocksize
parser.add_argument('-b', type=int, default=2000)
#How many blocks
parser.add_argument('-nblocks', type=int, default=14)
#How many trials to run
parser.add_argument('-trials', type=int, default=1)
#What matrix file (.npy) to load
parser.add_argument('-matrix', default=None)
#Are the placements fixed by the user or determined by the scheduler?
parser.add_argument('-fixed', default=0, type=int)
#How many GPUs to run on?
parser.add_argument('-ngpus', default=4, type=int)
args = parser.parse_args()

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    print("CUDA_VISIBLE_DEVICES is not set. Assuming 0-3")
    cuda_visible_devices = list(range(4))
else:
    cuda_visible_devices = cuda_visible_devices.strip().split(',')
    cuda_visible_devices = list(map(int, cuda_visible_devices))

gpus = cuda_visible_devices[:args.ngpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

print("Time:", 3*np.random.rand())
#raise Exception:
#    print("Exception raised")

#ENV VARIABLES INHERIT FROM PARENT. CAN'T CHANGE IT WITHIN
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
