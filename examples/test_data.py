# Sample input data used by conjugate gradient
# and preconditioned conjugate gradient examples.

import numpy as np
from scipy import sparse as sp
#from numba import jit

#@jit
def laplacian_helper(n, offset, num_edges, data, indices, indptr):
    indptr[0] = 0
    current_indptr = 1
    current_index = 0
    for i in range(n):
        if i >= offset:
            data[current_index] = 1.
            indices[current_index] = i - offset
            current_index += 1
        if i >= 1 and i % offset:
            data[current_index] = 1.
            indices[current_index] = i - 1
            current_index += 1
        data[current_index] = -4.
        indices[current_index] = i
        current_index += 1
        if i < n - 1 and (i + 1) % offset:
            data[current_index] = 1.
            indices[current_index] = i + 1
            current_index += 1
        if i < n - offset:
            data[current_index] = 1.
            indices[current_index] = i + offset
            current_index += 1
        indptr[current_indptr] = current_index
        current_indptr += 1
    assert current_indptr == n + 1
    assert current_index == num_edges 

# Generate sparse matrix for testing.
# This is the matrix representation of
# a 5 point stencil on a 2D grid.
def discrete_laplacian(n):
    assert n > 0
    offset = int(np.sqrt(n))
    num_edges = n + 2 * (n - 1 - (n - 1) // offset) + 2 * (n - offset)
    indptr = np.empty(n + 1, np.uint64)
    indices = np.empty(num_edges, np.uint64)
    data = np.empty(num_edges)
    laplacian_helper(n, offset, num_edges, data, indices, indptr)
    return sp.csc_matrix((data, indices, indptr), shape=(n, n))

#@jit
def laplacian_3d_helper(n, inner_offset, outer_offset, num_edges, data, indices, indptr):
    indptr[0] = 0
    current_indptr = 1
    current_index = 0
    for i in range(n):
        if i >= outer_offset:
            data[current_index] = 1.
            indices[current_index] = i - outer_offset
            current_index += 1
        if i >= inner_offset and (i // inner_offset) % inner_offset:
            data[current_index] = 1.
            indices[current_index] = i - inner_offset
            current_index += 1
        if i and i % inner_offset:
            data[current_index] = 1.
            indices[current_index] = i - 1
            current_index += 1
        data[current_index] = -6.
        indices[current_index] = i
        current_index += 1
        if i + 1 < n and (i + 1) % inner_offset:
            data[current_index] = 1.
            indices[current_index] = i + 1
            current_index += 1
        if i < n - inner_offset and (i // inner_offset + 1) % inner_offset:
            data[current_index] = 1.
            indices[current_index] = i + inner_offset
            current_index += 1
        if i < n - outer_offset:
            data[current_index] = 1.
            indices[current_index] = i + outer_offset
            current_index += 1
        indptr[current_indptr] = current_index
        current_indptr += 1
    assert current_indptr == n + 1
    assert current_index == num_edges

# Same, but for a 3d grid.
def discrete_laplacian_3d(n):
    assert n > 0
    inner_offset = int(np.cbrt(n))
    outer_offset = inner_offset**2
    num_edges = n + 2 * (n - 1 - (n - 1) // inner_offset)
    num_whole_diagonal_blocks = n // outer_offset
    last_block_partial = n % outer_offset
    num_edges = n
    num_edges += 2 * (n - 1 - (n - 1) // inner_offset)
    num_edges += 2 * (num_whole_diagonal_blocks * (inner_offset - 1) * inner_offset)
    if last_block_partial > inner_offset:
        num_edges += 2 * (last_block_partial - inner_offset)
    num_edges += 2 * (n - outer_offset)
    indptr = np.empty(n + 1, np.uint64)
    indices = np.empty(num_edges, np.uint64)
    data = np.empty(num_edges)
    laplacian_3d_helper(n, inner_offset, outer_offset, num_edges, data, indices, indptr)
    return sp.csc_matrix((data, indices, indptr), shape=(n, n))
