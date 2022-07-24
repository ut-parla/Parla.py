import os
import sys
import numpy as np

cuda_visible_devices =  os.environ.get('CUDA_VISIBLE_DEVICES').split(',')
cuda_visible_devices = [int(i) for i in cuda_visible_devices]

if cuda_visible_devices is None:
    print("Warning CUDA_VISIBLE_DEVICES is not set.")
    cuda_visible_devices = list(range(4))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)

print("Time:", 3*np.random.rand())
#raise Exception:
#    print("Exception raised")

#ENV VARIABLES INHERIT FROM PARENT. CAN'T CHANGE IT WITHIN
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
