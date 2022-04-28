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

def generate_import_list(nCPU, nGPU):
	modules = []
	for i in range(0, nCPU):
		modules.append(generate_name('cpu', i))

	for j in range(0, nGPU):
		modules.append(generate_name('gpu', j))

	return modules

def generate_instances(modules):
	instances = []
	nmodules = len(modules)
	for i in range(0,nmodules):
		if modules[i] in sys.modules:
			del sys.modules[modules[i]]
		module = importlib.import_module(modules[i])
		instances.append(sys.modules[modules[i]])
	return instances

@contextmanager
def load_inst(dev):
	save = sys.modules[this.modules[dev]]
	sys.modules[this.modules[dev]] = this.instances[dev]
	yield this.instances[dev]
	sys.modules[this.modules[dev]] = save

def initialize():
	assert len(this.modules) == len(this.instances)	
	for i in range(0,len(this.modules)):
		with load_inst(i) as spec:
			if i == 0:
				spec.start()
			else:
				spec.start(i-1)

def finalize():
	assert len(this.modules) == len(this.instances)	
	for i in range(0,len(this.modules)):
		with load_inst(i) as spec:
			spec.end()

def getDeviceIndex(array):
	if isinstance(array, (np.ndarray, np.generic)):
		return 0
	else:
		return array.data.device.id + 1	

def reduction(array):
	result = None
	dev = getDeviceIndex(array)
	with load_inst(dev) as spec:
		result = spec.reduction(array)
	return result
