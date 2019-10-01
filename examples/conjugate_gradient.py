# ILU(0) preconditioned conjogate gradient
# See Algorithm 9.1 of Iterative Methods for Sparse Linear Systems.

import numpy as np
import scipy.sparse as sp

from test_data import discrete_laplacian, discrete_laplacian_3d

# Note: this example uses CSC format throughout.

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

def main():
    n = 52488
    A = discrete_laplacian(n)
    b = np.ones(n)
    x = np.ones(n)
    x, residual = cg(A, x, b, maxiters=800)
    #print(b - A * x)
    assert(np.linalg.norm(residual, ord=np.inf) < 1E-8)

if __name__ == '__main__':
    main()
    print("success")
