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
uint64_t = nb.uint64
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
    #assert 0 <= i0 < n0
    #assert 0 <= i1 < n1
    #assert 0 <= i2 < n2
    #assert 0 <= i3 < n3
    return i0, i1, i2, i3

# Sequential preprocessing run on CPU to sort work items by DAG level.
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
    ldc = num_dirs # doesn't matter since there's only one column here too.
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

@cuda.jit(nb.void(uint_t[:], float_t[:,:,:,:,:], float_t[:], float_t[:,:,:,:], float_t[:], float_t[:,:], float_t, uint_t[:], float_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t))
def compute_fluxes(work_items, I, I_flat, sigma, sigma_flat, directions, sigma_a_s, tgroup_id, num_dirs_inv, x_offset, y_offset, z_offset, frequency_offset, sigma_I_offset, sigma_I_x_mul, direction_correction):
    block_base_index = cuda.shared.array((1,), uint_t)
    if not cuda.threadIdx.x:
        block_base_index[0] = cuda.atomic.add(tgroup_id, 0, 1) * uint_t(cuda.blockDim.x)
    cuda.syncthreads()
    work_item_idx = block_base_index[0] + uint_t(cuda.threadIdx.x)
    if work_item_idx >= work_items.shape[0]:
        return
    idx = work_items[work_item_idx]
    sigma_flat_idx = idx
    # TODO: The uint_t specs for nx, ny, and nz here drastically change the result. WHY?
    nx = uint_t(sigma.shape[0])
    ny = uint_t(sigma.shape[1])
    nz = uint_t(sigma.shape[2])
    num_dirs = uint_t(sigma.shape[3])
    num_groups = uint_t(I.shape[4])
    sigma_x, sigma_y, sigma_z, dir_idx = unravel_4d_index(nx, ny, nz, directions.shape[0], idx)
    I_flat_idx = sigma_flat_idx * num_groups + sigma_I_x_mul * sigma_x + z_offset * sigma_y + sigma_I_offset - direction_correction * dir_idx
    # Now change abstract indices in the iteration space into indices into I.
    # This is necessary since the beginning and end of each spatial axis
    # is used for boundary conditions.
    ix = sigma_x + uint_t(1)
    iy = sigma_y + uint_t(1)
    iz = sigma_z + uint_t(1)
    dirx = directions[dir_idx, 0]
    diry = directions[dir_idx, 1]
    dirz = directions[dir_idx, 2]
    x_has_sign = dirx < float_t(0.)
    y_has_sign = diry < float_t(0.)
    z_has_sign = dirz < float_t(0.)
    x_neighbor_idx = ix + uint_t(1) if x_has_sign else ix - uint_t(1)
    y_neighbor_idx = iy + uint_t(1) if y_has_sign else iy - uint_t(1)
    z_neighbor_idx = iz + uint_t(1) if z_has_sign else iz - uint_t(1)
    x_neighbor_flat_idx = I_flat_idx + x_offset if x_has_sign else I_flat_idx - x_offset
    y_neighbor_flat_idx = I_flat_idx + y_offset if y_has_sign else I_flat_idx - y_offset
    z_neighbor_flat_idx = I_flat_idx + z_offset if z_has_sign else I_flat_idx - z_offset
    x_coef = -float_t(nx) if x_has_sign else float_t(nx)
    y_coef = -float_t(ny) if y_has_sign else float_t(ny)
    z_coef = -float_t(nz) if z_has_sign else float_t(nz)
    denominator = (sigma_a_s -
                   x_coef * dirx -
                   y_coef * diry -
                   z_coef * dirz)
    # In full-blown versions of this code this sum is actually an inner product
    # that uses coefficients specific to this direction. Frequencies may also be
    # considered, but some kind of lower-dimensional thing is usually used
    # to store the scattering terms. This sum just runs over the directions.
    incoming_scattering = float_t(0.)
    sigma_dir_block_idx = sigma_flat_idx - dir_idx
    for j64 in range(uint_t(sigma.shape[3])):
        j = uint_t(j64)
        #incoming_scattering += sigma[sigma_x, sigma_y, sigma_z, j]
        incoming_scattering += sigma_flat[sigma_dir_block_idx + j]
    incoming_scattering *= num_dirs_inv
    # For simplicity we're assuming all frequencies scatter the same, so
    # sum across frequencies now.
    x_factor = x_coef * dirx
    y_factor = y_coef * diry
    z_factor = z_coef * dirz
    div = float_t(1.) / denominator
    x_neighbor_flat_idx_last = x_neighbor_flat_idx + (num_groups - 1) * num_dirs
    y_neighbor_flat_idx_last = y_neighbor_flat_idx + (num_groups - 1) * num_dirs
    z_neighbor_flat_idx_last = z_neighbor_flat_idx + (num_groups - 1) * num_dirs
    while True:
        cuda.threadfence()
        # Stop if the upstream neighbors aren't ready.
        if (math.isnan(I[x_neighbor_idx, iy, iz, dir_idx, -1]) or
            math.isnan(I[ix, y_neighbor_idx, iz, dir_idx, -1]) or
            math.isnan(I[ix, iy, z_neighbor_idx, dir_idx, -1])):
            continue
        #if (math.isnan(I_flat[x_neighbor_flat_idx_last]) or
        #    math.isnan(I_flat[y_neighbor_flat_idx_last]) or
        #    math.isnan(I_flat[z_neighbor_flat_idx_last])):
        #    continue
        # For simplicity we're assuming all frequencies scatter the same, so
        # sum across frequencies now.
        for k64 in range(uint_t(I.shape[4])):
            k = uint_t(k64)
            numerator = (incoming_scattering -
                         x_factor * I[x_neighbor_idx,iy,iz,dir_idx,k] -
                         y_factor * I[ix,y_neighbor_idx,iz,dir_idx,k] -
                         z_factor * I[ix,iy,z_neighbor_idx,dir_idx,k])
            #numerator = (incoming_scattering - 
            #             x_factor * I_flat[x_neighbor_flat_idx + k * num_dirs] -
            #             y_factor * I_flat[y_neighbor_flat_idx + k * num_dirs] -
            #             z_factor * I_flat[z_neighbor_flat_idx + k * num_dirs])
            flux = numerator * div
            if k == I.shape[4] - uint_t(1):
                cuda.threadfence()
            I[ix,iy,iz,dir_idx,k] = flux
            #I_flat[I_flat_idx] = flux
        break

def sweep_step(work_items, tgroup_id, I, sigma, new_sigma, coefs, directions, sigma_a, sigma_s):
    I[1:-1,1:-1,1:-1,:,-1] = np.nan
    tgroup_id[0] = 0
    # Sweep across the graph for the differencing scheme for the gradient.
    chunk_size = 1024
    num_blocks = (work_items.shape[0] + chunk_size - 1) // chunk_size
    assert I.strides[3] == 4
    I_flat = np.swapaxes(I, 3, 4).ravel()
    sigma_flat = sigma.ravel()
    x_offset = I.shape[1] * I.shape[2] * I.shape[3] * I.shape[4]
    y_offset = I.shape[2] * I.shape[3] * I.shape[4]
    z_offset = I.shape[3] * I.shape[4]
    # direction_offset = 1
    sigma_I_offset = (I.shape[1] * I.shape[2] + I.shape[2] + 1) * z_offset
    sigma_I_x_mul = (I.shape[1] + I.shape[2] + 1) * z_offset
    direction_correction = (I.shape[4] - 1)
    frequency_offset = I.shape[3]
    cp.cuda.get_current_stream().synchronize()
    start = perf_counter()
    compute_fluxes[num_blocks, chunk_size, 0, uint_t_nbytes](work_items, I, I_flat, sigma, sigma_flat, directions, sigma_a + sigma_s, tgroup_id, 1. / I.shape[1], x_offset, y_offset, z_offset, frequency_offset, sigma_I_offset, sigma_I_x_mul, direction_correction)
    cp.cuda.get_current_stream().synchronize()
    stop = perf_counter()
    print("sweep kernel time:", stop - start)
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
    nx = ny = nz = uint_t(64)
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
