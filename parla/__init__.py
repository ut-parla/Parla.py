"""
Parla is a parallel language for orchestrating high-performance array-based programs.
"Orchestration" refers to controlling lower-level operations from a higher-level language.
In this case, Parla orchestrates low-level array operations and other existing high-performance operations (written in C, C++, or FORTRAN).
"""

# For now, with the prototype, force underlying BLAS libs to run sequentially.
import os

__all__ = []

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
