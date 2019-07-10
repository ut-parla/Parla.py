from parla.tasks import *

assert False

"""
This is not actually runnable. Think of it as very detailed pseudocode.
We will make it run soon-ish. Both library changes and code changes here will be needed.
"""

loc = {
    (0, 0): gpu(0),
    (0, 1): gpu(1),
    (1, 0): gpu(2),
    (1, 1): gpu(3),
    }
# Eventually: loc = infer_placements()
# Or even better: loc = abstract_cartesian(n, n)
def mem(i, j):
    return loc((i,j)).memory()

# TODO: Compare to how this is done in HPF (High-Performance Fortran).
#  Alignment: an abstract set of storage locations and then placenebt of data in those abstract locations.
#  Distribution (at runtime): associate abstract locations to physical locations. This mapping was static.
#
# (Reference Pingali 1989)

# Examples:
#  Fox's Algorithm: Collective regular communication (broadcast and reduce)
#  Cholesky: Less regular data movement and task placement, more complex dependencies.


def fox(y: Array[1, Array[1, int]], A: Array[2, Array[2, int]], x: Array[1, Array[1, int]]):
    """y = Ax

    y, A, and x are pre-blocked and passed in as arrays of arrays to
    allow each sub-array to be in a different location. (This
    sub-array structure could be eliminated by allowing arrays to be
    partitioned natively.)

    """

    # n is the size of the block arrays
    n = y.indicies[-1]

    assert x.indicies[-1] == n
    assert A.indicies[-1] == (n, n)

    assert all(y[i].memory_location == mem(i, i) for i in range(0, n))
    assert all(x[i].memory_location == mem(i, i) for i in range(0, n))
    assert all(A[i, j].memory_location == mem(i, j) for i in range(0, n) for j in range(0, n))

    xp = Array[2].with_dims(n,n)
    yp = Array[2].with_dims(n,n)

    # broadcast along columns
    for j in range(0, n): # columns
        for i in range(0, n): # rows
            @spawn(B[i, j], placement=loc(i, j))
            def b():
                xp[i, j] = mem(i, j)(x[i])

    # block-wise multiplication
    for i in range(0, n): # rows
        for j in range(0, n): # columns
            @spawn(M[i, j], [B[i, j]], placement=loc(i, j))
            def m():
                yp[i, j] = A[i, j] @ xp[i, j]

    # reduce along rows
    for i in range(0, n): # rows
        @spawn(R[i], [M[i, 0:n]], placement=loc(i, i))
        def r():
            y[i][:] = 0
            for j in range(0, n): # columns
                t = mem(i, i)(yp[i, j])
                y[i] = y[i] + t

    # join the reduce tasks
    @spawn(None, [R[0:n]])
    def done():
        pass
    return done # Return the join task
