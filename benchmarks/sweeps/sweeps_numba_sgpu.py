import cupy as cp
import math
#from matplotlib import pyplot as plt
import numba as nb
from numba import cuda
import numpy as np
from time import perf_counter

# TODO: switch to sigma being per-direction since atomics won't be needed that way.
# That's more representative of how real codes that do this run anyway.

uint_t = nb.uint32
uint_t_npy = np.uint32
uint_t_nbytes = 4
float_t = nb.float32
float_t_npy = np.float32

# A hand-rolled version of ravel_multi_index for use inside a jitted function.
# This is special cased for 4D indexing since that's what we need.
@nb.jit(uint_t(uint_t, uint_t, uint_t, uint_t, uint_t, uint_t, uint_t, uint_t), nopython = True)
def ravel_4d_index(n0, n1, n2, n3, i0, i1, i2, i3):
    assert 0 <= i0 < n0
    assert 0 <= i1 < n1
    assert 0 <= i2 < n2
    assert 0 <= i3 < n3
    return i3 + i2 * n3 + i1 * n2 * n3 + i0 * n1 * n2 * n3

# This is only used in the CPU preprocessing code that organizes the work items.
@cuda.jit(nb.types.UniTuple(uint_t, 4)(uint_t, uint_t, uint_t, uint_t, uint_t), device = True)
def unravel_4d_index(n0, n1, n2, n3, ix):
    i3 = uint_t(ix % n3)
    d2 = uint_t(ix // n3)
    i2 = uint_t(d2 % n2)
    d1 = uint_t(d2 // n2)
    i1 = uint_t(d1 % n1)
    d0 = uint_t(d1 // n1)
    i0 = uint_t(d0 % n0)
    assert 0 <= i0 < n0
    assert 0 <= i1 < n1
    assert 0 <= i2 < n2
    assert 0 <= i3 < n3
    return i0, i1, i2, i3

# Sequential preprocessing run on CPU.
@nb.jit(uint_t[:](float_t[:,:], uint_t, uint_t, uint_t, uint_t), nopython = True)
def generate_items(directions, nx, ny, nz, nd):
    items = np.empty((nx * ny * nz * nd), uint_t_npy)
    items_index = 0
    for wavefront in range(nx + ny + nz):
        # TODO: move this loop in past the spatial indices.
        # Group directions by octant.
        for dir_idx in range(nd):
            x_has_sign = np.signbit(directions[dir_idx, 0])
            y_has_sign = np.signbit(directions[dir_idx, 1])
            z_has_sign = np.signbit(directions[dir_idx, 2])
            for x_offset in range(min(wavefront + 1, nx)):
                ix = nx - x_offset - 1 if x_has_sign else x_offset
                for y_offset in range(max(0, wavefront - x_offset - nz + 1),
                                      min(wavefront - x_offset + 1, ny)):
                    iy = ny - y_offset - 1 if y_has_sign else y_offset
                    z_offset = wavefront - x_offset - y_offset
                    iz = nz - z_offset - 1 if z_has_sign else z_offset
                    assert(wavefront == x_offset + y_offset + z_offset)
                    assert(0 <= ix < nx and 0 <= iy <ny and 0 <= iz < nz)
                    items[items_index] = ravel_4d_index(nx, ny, nz, nd, ix, iy, iz, dir_idx)
                    items_index += 1
    return items

def compute_new_scattering(sigma_s, I, coefs, new_sigma):
    num_dirs = I.shape[3]
    num_groups = I.shape[4]
    # TODO: Switch this over to scattering only between directions (like tycho 2 does) instead of being only between frequencies.
    # TODO: Turn this into a cublas sgemmBatched call since cupy probably doesn't even call the right routine for this.
    assert I.strides[3] == 4
    assert new_sigma.strides[3] == 4
    cublas_handle = cp.cuda.device.get_cublas_handle()
    transa = cp.cuda.cublas.CUBLAS_OP_N
    transb = cp.cuda.cublas.CUBLAS_OP_N
    m = num_dirs
    n = 1
    k = num_groups
    alpha_block = np.array(1., 'f')
    alpha = alpha_block.__array_interface__['data'][0]
    lda = num_dirs
    strideA = num_groups * num_dirs
    B = coefs.data.ptr
    ldb = num_groups # doesn't actually matter since there's only one column.
    strideB = 0
    beta_block = np.array(0., 'f')
    beta = beta_block.__array_interface__['data'][0]
    ldc = num_dirs
    strideC = num_dirs
    batchCount = I.shape[2] - 2
    for i in range(1, I.shape[0] - 1):
        for j in range(1, I.shape[1] - 1):
            A_base = I[i,j,1:-1]
            A = A_base.data.ptr
            C_base = new_sigma[i-1,j-1]
            C = C_base.data.ptr
            cp.cuda.cublas.sgemmStridedBatched(cublas_handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    #new_sigma[:] = I[1:-1,1:-1,1:-1] @ coefs

@cuda.jit(nb.void(uint_t[:], float_t[:,:,:,:,:], float_t[:,:,:,:], float_t[:,:], float_t, float_t, uint_t[:]))
def compute_fluxes(work_items, I, sigma, directions, sigma_a, sigma_s, tgroup_id):
    block_base_index = cuda.shared.array((1,), uint_t)
    if cuda.threadIdx.x == 0:
        block_base_index[0] = cuda.atomic.add(tgroup_id, 0, 1) * cuda.blockDim.x
    cuda.syncthreads()
    work_item_idx = block_base_index[0] + cuda.threadIdx.x
    if work_item_idx >= work_items.shape[0]:
        return
    idx = work_items[work_item_idx]
    # TODO: The uint_t specs for nx, ny, and nz here drastically change the result. WHY?
    nx = uint_t(sigma.shape[0])
    ny = uint_t(sigma.shape[1])
    nz = uint_t(sigma.shape[2])
    num_dirs = sigma.shape[3]#uint_t(I.shape[3])
    num_groups = I.shape[4]#uint_t(I.shape[4])
    sigma_x, sigma_y, sigma_z, dir_idx = unravel_4d_index(nx, ny, nz, directions.shape[0], idx)
    # Now change abstract indices in the iteration space into indices into I.
    # This is necessary since the beginning and end of each spatial axis
    # is used for boundary conditions.
    ix = uint_t(sigma_x + 1)
    iy = uint_t(sigma_y + 1)
    iz = uint_t(sigma_z + 1)
    # TODO: just pull the directions values into registers in advance.
    x_has_sign = directions[dir_idx, 0] < 0.
    y_has_sign = directions[dir_idx, 1] < 0.
    z_has_sign = directions[dir_idx, 2] < 0.
    x_neighbor_idx = uint_t(ix + 1 if x_has_sign else ix - 1)
    y_neighbor_idx = uint_t(iy + 1 if y_has_sign else iy - 1)
    z_neighbor_idx = uint_t(iz + 1 if z_has_sign else iz - 1)
    x_coef = -nx if x_has_sign else nx
    y_coef = -ny if y_has_sign else ny
    z_coef = -nz if z_has_sign else nz
    denominator = (sigma_a + sigma_s -
                   x_coef * directions[dir_idx, 0] -
                   y_coef * directions[dir_idx, 1] -
                   z_coef * directions[dir_idx, 2])
    # In full-blown versions of this code this sum is actually an inner product
    # that uses coefficients specific to this direction. Frequencies may also be
    # considered, but some kind of lower-dimensional thing is usually used
    # to store the scattering terms. This sum just runs over the directions.
    incoming_scattering = 0.
    for j in range(sigma.shape[3]):
        incoming_scattering += sigma[sigma_x, sigma_y, sigma_z, j]
    incoming_scattering /= num_dirs
    # For simplicity we're assuming all frequencies scatter the same, so
    # sum across frequencies now.
    x_factor = x_coef * directions[dir_idx, 0]
    y_factor = y_coef * directions[dir_idx, 1]
    z_factor = z_coef * directions[dir_idx, 2]
    div = 1. / denominator
    denominator = (sigma_a + sigma_s -
                   x_coef * directions[dir_idx, 0] -
                   y_coef * directions[dir_idx, 1] -
                   z_coef * directions[dir_idx, 2])
    while True:
        cuda.threadfence()
        # Stop if the upstream neighbors aren't ready.
        if (math.isnan(I[x_neighbor_idx, iy, iz, dir_idx, -1]) or
            math.isnan(I[ix, y_neighbor_idx, iz, dir_idx, -1]) or
            math.isnan(I[ix, iy, z_neighbor_idx, dir_idx, -1])):
            continue
        # For simplicity we're assuming all frequencies scatter the same, so
        # sum across frequencies now.
        for k in range(I.shape[4]):
            numerator = (incoming_scattering -
                         x_factor * I[x_neighbor_idx,iy,iz,dir_idx,k] -
                         y_factor * I[ix,y_neighbor_idx,iz,dir_idx,k] -
                         z_factor * I[ix,iy,z_neighbor_idx,dir_idx,k])
            flux = numerator * div
            if k == I.shape[4] - 1:
                cuda.threadfence()
            I[ix,iy,iz,dir_idx,k] = flux
        break

def sweep_step(work_items, tgroup_id, I, sigma, new_sigma, coefs, directions, sigma_a, sigma_s):
    I[1:-1,1:-1,1:-1,:,-1] = np.nan
    tgroup_id[0] = 0
    # Sweep across the graph for the differencing scheme for the gradient.
    chunk_size = 1024
    num_blocks = (work_items.shape[0] + chunk_size - 1) // chunk_size
    print(I.shape, sigma.shape)
    compute_fluxes[num_blocks, chunk_size, 0, uint_t_nbytes](work_items, I, sigma, directions, sigma_a, sigma_s, tgroup_id)
    # Compute the scattering terms in the collision operator.
    compute_new_scattering(sigma_s, I, coefs, new_sigma)

def partition_sphere(num_vert_dirs, num_horiz_dirs):
    horiz_centers = np.linspace(0, 2 * np.pi, num_horiz_dirs, endpoint=False) + np.pi / num_horiz_dirs
    vert_centers = np.arcsin(np.linspace(-1, 1, num_vert_dirs, endpoint = False) + 1. / num_vert_dirs)
    dirs = np.empty((num_vert_dirs, num_horiz_dirs, 3), float_t_npy)
    dirs[:,:,0] = np.cos(vert_centers)[...,None] * np.cos(horiz_centers)
    dirs[:,:,1] = np.cos(vert_centers)[...,None] * np.sin(horiz_centers)
    dirs[:,:,2] = np.sin(vert_centers)[...,None]
    return dirs.reshape((num_vert_dirs * num_horiz_dirs, 3))

def get_positive_x_direction(directions):
    return directions[:,0].argmax()

# TODO: NVProf

def sweep():
    nx = ny = nz = uint_t(32)
    num_vert_dirs = uint_t(10)
    num_horiz_dirs = uint_t(20)
    num_groups = uint_t(32)
    num_iters = 1
    pulse_strength = 1.
    sigma_a = .01
    sigma_s = .95
    num_dirs = num_vert_dirs * num_horiz_dirs
    # Make directions the smallest stride to get coalesced acceses.
    I = np.swapaxes(cp.zeros((nx + 2, ny + 2, nz + 2, num_groups, num_dirs), float_t_npy), -1, -2)
    sigma = cp.zeros_like(I[1:-1, 1:-1, 1:-1, :, 0])
    new_sigma = cp.zeros_like(sigma)
    directions = partition_sphere(num_vert_dirs, num_horiz_dirs)
    assert(directions.shape[0] == I.shape[3])
    positive_x_dir = get_positive_x_direction(directions)
    I[0,ny//2+1,nz//2+1,positive_x_dir, 0] = pulse_strength
    live_ix = 0
    live_iy = ny // 2
    live_iz = nz // 2
    live_dir = positive_x_dir
    work_items = generate_items(directions, nx, ny, nz, uint_t(directions.shape[0]))
    work_items = cp.array(work_items)
    directions = cp.array(directions)
    # TODO: tune block sizes on a newer GPU.
    # TODO: estimate number of blocks based off of problem size.
    tgroup_id = cp.zeros((1,), uint_t_npy)
    coefs = cp.empty((num_groups,), np.float32)
    coefs[:] = sigma_s / num_groups
    start = perf_counter()
    for i in range(num_iters):
        sigma, new_sigma = new_sigma, sigma
        sweep_step(work_items, tgroup_id, I, sigma, new_sigma, coefs, directions, sigma_a, sigma_s)
    stop = perf_counter()
    print("I sum:", I.sum())
    print("new sigma sum:", new_sigma.sum())
    print("total time:", stop - start)
    # Normalize so the results are roughly the same as the previous version.
    new_sigma /= num_dirs
    new_sigma = new_sigma.get()
    #plt.imshow(new_sigma[0,:,:].sum(axis=-1))#, vmin=0, vmax=1)
    #plt.colorbar()
    #plt.show()
    #plt.imshow(new_sigma[-3,:,:].sum(axis=-1))#, vmin=0, vmax=1)
    #plt.colorbar()
    #plt.show()
    #plt.imshow(new_sigma[:,live_iy,:].sum(axis=-1))#, vmin=0, vmax=1)
    #plt.colorbar()
    #plt.show()
    #plt.imshow(new_sigma[:,:,live_iz].sum(axis=-1))#, vmin=0, vmax=1)
    #plt.colorbar()
    #plt.show()

#print(compute_fluxes.inspect_asm())
#print(compute_fluxes.inspect_types())
sweep()
#cuda.profile_stop()
