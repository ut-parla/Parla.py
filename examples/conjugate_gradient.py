# ILU(0) preconditioned conjogate gradient
# See Algorithm 9.1 of Iterative Methods for Sparse Linear Systems.

import numpy as np
from scipy import linalg as la
import scipy.sparse as ss
import scipy.sparse.linalg as sla

# Note: this example uses CSC format throughout.

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
    indptr[0] = 0
    current_indptr = 1
    current_index = 0
    for i in range(n):
        if i >= offset:
            data[current_index] = -1.
            indices[current_index] = i - offset
            current_index += 1
        if i >= 1 and i % offset:
            data[current_index] = -1.
            indices[current_index] = i - 1
            current_index += 1
        data[current_index] = 16
        indices[current_index] = i
        current_index += 1
        if i < n - 1 and (i + 1) % offset:
            data[current_index] = -1.
            indices[current_index] = i + 1
            current_index += 1
        if i < n - offset:
            data[current_index] = -1.
            indices[current_index] = i + offset
            current_index += 1
        indptr[current_indptr] = current_index
        current_indptr += 1
    assert current_indptr == n + 1
    assert current_index == num_edges
    return ss.csc_matrix((data, indices, indptr), shape=(n, n))

def cg(A, x_initial, b, tol=1E-8, maxiters=100):
    """
    Solve A @ x = b for x (where A is symmetric and positive definite).

    :param A: A symmetric positive definite CSC or CSR matrix.
    :param x_initial: The initial guess of x. Overwritten with the result.
    :param b: A vector.
    :param tol: The tolerance for convergence detection.
    :param maxiters: The maximum number of iterations.
    :return: a pair of the final x and the error vector r.
    """
    # The code here is written in terms of a CSC matrix, however,
    # since the input matrix must be symmetric, everything will
    # still work if the input matrix is in CSR format instead.
    x = x_initial
    r = b - A * x
    p = r.copy()
    r_norm2 = np.inner(r, r)
    for j in range(maxiters):
        Ap = A * p

        alpha = r_norm2 / np.inner(Ap, p)
        x += alpha * p
        r -= alpha * Ap
        # Just use inf. norm for convergence (for simplicity)
        err = np.linalg.norm(r, ord=np.inf)
        print(err)
        if err < tol:
            return x, r
        r_norm2_new = np.inner(r, r)
        p = r + (r_norm2_new / r_norm2) * p
        r_norm2 = r_norm2_new
    return x, r

def test_cg():
    n = 64
    A = discrete_laplacian(n)
    b = np.ones(n)
    x = np.ones(n)
    x, residual = cg(A, x, b)
    # print(b - A * x)
    assert(np.linalg.norm(residual, ord=np.inf) < 1E-8)

if __name__ == '__main__':
    test_cg()
    print("success")
