import cupy as cp
import numpy as np
from sys import argv

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=28000, help="Size of the matrix")
args = parser.parse_args()


n = args.n

np.random.seed(10)
a = np.random.rand(n, n)
a = a @ a.T

np.save(f"chol_{n}", a)
