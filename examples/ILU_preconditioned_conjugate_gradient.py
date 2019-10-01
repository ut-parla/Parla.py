# ILU(0) preconditioned conjogate gradient
# See Algorithm 9.1 of Iterative Methods for Sparse Linear Systems.

import numpy as np
import scipy.sparse as sp

from test_data import discrete_laplacian, discrete_laplacian_3d

# Note: this example uses CSC format throughout.

# Placeholder serial routine to be replaced by something better optimized.
# Note, with this loop order on CSC matrices, the dependencies follow the
# sparse matrix topology. The dependency structure for other layouts
def ILU0_csc(A):
    indptr = A.indptr
    indices = A.indices
    data = A.data
    for node in range(A.shape[0]):
        parents = indices[indptr[node]:indptr[node + 1]]
        parents_data = data[indptr[node]:indptr[node + 1]]
        for parent_local_index in range(len(parents)):
            parent_index = parents[parent_local_index]
            if parent_index < node:
                grandparents = indices[indptr[parent_index]:indptr[parent_index + 1]]
                grandparents_data = data[indptr[parent_index]:indptr[parent_index + 1]]
                for grandparent_local_index in range(len(grandparents)):
                    grandparent_index = grandparents[grandparent_local_index]
                    if grandparent_index <= parent_index:
                        # Could jump to start position if sorted.
                        continue
                    if A[grandparent_index, node]:
                        A[grandparent_index, node] -= grandparents_data[grandparent_local_index] * parents_data[parent_local_index]
            elif parent_index > node:
                parents_data[parent_local_index] /= A[node, node]

# Lower triangular solve
# Assume sorted edges.
def lower_triangular_solve_csc(a, b, diag_ones = False, overwrite_b = False):
    if overwrite_b:
        out = b
    else:
        out = b.copy()
    indptr = a.indptr
    indices = a.indices
    data = a.data
    for i in range(a.shape[0]):
        lower = indptr[i]
        upper = indptr[i+1]
        neighbors = indices[lower:upper]
        neighbor_data = data[lower:upper]
        start_index = np.searchsorted(neighbors, i)
        if not diag_ones:
            assert neighbors[start_index] == i
            # TODO: Better singularity check?
            assert neighbor_data[start_index]
            out[i] /= neighbor_data[start_index]
        for j in range(start_index + 1, neighbors.shape[0]):
            out[neighbors[j]] -= neighbor_data[j] * out[i]
    return out

def upper_triangular_solve_csc(a, b, diag_ones = False, overwrite_b = False):
    if overwrite_b:
        out = b
    else:
        out = b.copy()
    indptr = a.indptr
    indices = a.indices
    data = a.data
    for i in range(a.shape[0]-1, -1, -1):
        lower = indptr[i]
        upper = indptr[i+1]
        neighbors = indices[lower:upper]
        neighbor_data = data[lower:upper]
        end_index = np.searchsorted(neighbors, i)
        if not diag_ones:
            assert neighbors[end_index] == i
            # TODO: Better singularity check?
            assert neighbor_data[end_index]
            out[i] /= neighbor_data[end_index]
        for j in range(end_index):
            out[neighbors[j]] -= neighbor_data[j] * out[i]
    return out

def pcg(A, x_initial, b, tol=1E-8, maxiters=100):
    """
    Solve A @ x = b for x (where A is symmetric and positive definite).
    Use an incomplete LU factorization with no fill as a preconditioner.

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
    ilu = A.copy()
    ILU0_csc(ilu)
    x = x_initial
    r = b - A @ x
    z_part = upper_triangular_solve_csc(ilu, r)
    z = lower_triangular_solve_csc(ilu, z_part, diag_ones=True, overwrite_b = True)
    p = z.copy()
    rz_inner = np.inner(r, z)
    for j in range(maxiters):
        Ap = A @ p
        alpha = rz_inner / np.inner(Ap, p)
        x += alpha * p
        r -= alpha * Ap
        # Just use inf. norm for convergence (for simplicity)
        err = np.linalg.norm(r, ord=np.inf)
        print(err)
        if err < tol:
            return x, r
        z_part = upper_triangular_solve_csc(ilu, r)
        z = lower_triangular_solve_csc(ilu, z_part, diag_ones = True, overwrite_b = True)
        rz_inner_new = np.inner(r, z)
        p = z + (rz_inner_new / rz_inner) * p
        rz_inner = rz_inner_new
    return x, r

def test_pcg():
    n = 52488
    A = discrete_laplacian(n)
    b = np.ones(n)
    x = np.ones(n)
    x, residual = pcg(A, x, b, maxiters=300)
    assert(np.linalg.norm(residual, ord=np.inf) < 1E-5)

if __name__ == '__main__':
    test_pcg()
    print("success")
