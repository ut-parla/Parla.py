from contextlib import contextmanager
import importlib
import sys
import numpy as np
import kokkos
this = sys.modules[__name__]


def setup(nCPU, nGPU):
	this.modules = generate_import_list(nCPU, nGPU)
	this.instances = generate_instances(modules)

def generate_name(deviceType, devID):
	return 'kokkos.'+deviceType+str(devID)+'.core'

def generate_instances(modules):
	instances = []
	nmodules = len(modules)
	for i in range(0,nmodules):
		if modules[i] in sys.modules:
			del sys.modules[modules[i]]
		module = importlib.import_module(modules[i])
		instances.append(sys.modules[modules[i]])
	return instances

def initialize():
    if i == 0:
        spec.start()
    else:
        spec.start(i-1)

def finalize():
    spec.end()

def reduction(array):
    result = None
    result = spec.reduction(array)
    return result
