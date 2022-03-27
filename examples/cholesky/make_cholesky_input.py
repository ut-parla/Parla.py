import cupy as cp
import numpy as np
from sys import argv

n = int(argv[1])

np.random.seed(10)
a = np.random.rand(n, n)
a = a @ a.T

np.save(f"chol_{n}", a)
