/*
 *  Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA CORPORATION nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 *  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @file
 * @brief This file contains all of cudaLibMg's public function declarations.
 */
#ifndef CUDALIBMG_H
#define CUDALIBMG_H

#define CUDALIBMG_MAJOR 0
#define CUDALIBMG_MINOR 1
#define CUDALIBMG_PATCH 0

#include <cstdint>
#include <cuda_runtime.h>
    
/**
 * \brief This enum contains all possible retun status of cudaLibMg.
 */ 
typedef enum
{
    CUDALIBMG_STATUS_SUCCESS,
    CUDALIBMG_STATUS_NOT_INITIALIZED,
    CUDALIBMG_STATUS_ALLOC_FAILED,
    CUDALIBMG_STATUS_INVALID_VALUE,
    CUDALIBMG_STATUS_ARCH_MISMATCH,
    CUDALIBMG_STATUS_MAPPING_ERROR,
    CUDALIBMG_STATUS_EXECUTION_FAILED,
    CUDALIBMG_STATUS_NOT_SUPPORTED,
    CUDALIBMG_STATUS_INTERNAL_ERROR,
    CUDALIBMG_STATUS_CUDA_ERROR,
    CUDALIBMG_STATUS_CUBLAS_ERROR,
    CUDALIBMG_STATUS_NCCL_ERROR,
} cudaLibMgStatus_t;

/**                                                                                                                     
 * \beief This enum decides how 1D device Ids (or process ranks) get mapped to a 2D grid.                               
 */
enum cudaLibMgGridMapping_t
{
    CUDALIBMG_GRID_MAPPING_ROW_MAJOR,
    CUDALIBMG_GRID_MAPPING_COL_MAJOR
};

/* Opaque structure storing the information for a dense, distributed matrix */
typedef void * cudaLibMgMatrixDesc_t;

/* Opaque structure storing the information of a processing grid */
typedef void * cudaLibMgGrid_t;

extern "C"
{
    /**
     * \brief Returns Version number of the cudaLibMg library.
     */
    size_t cudaLibMgGetVersion();

    /**
     * \brief Returns version number of the CUDA runtime that cuBLASMg was compiled against.
     */
    size_t cudaLibMgGetCudartVersion();

    /**
     * \brief Allocates resources related to the distributed matrix descriptor.
     * \param[out] desc the opaque data strcuture that holds the matrix descriptor
     * \param[in] numRows number of total rows
     * \param[in] numCols number of total columns
     * \param[in] rowBlockSize row block size
     * \param[in] colBlockSize column block size
     * \param[in] dataType the data type of each element in cudaDataType_t
     * \param[in] grid the opaque data structure of the device grid
     * \returns the status code
     */
    cudaLibMgStatus_t cudaLibMgCreateMatrixDesc(cudaLibMgMatrixDesc_t * desc, 
            int64_t numRows, int64_t numCols, int64_t rowBlockSize, int64_t colBlockSize,
            cudaDataType_t dataType, const cudaLibMgGrid_t grid);

    /**
     * \brief Releases the allocated resources related to the distributed grid.
     * \param[in] grid the opaque data strcuture that holds the distributed grid
     * \returns the status code
     */
    cudaLibMgStatus_t cudaLibMgDestroyMatrixDesc(cudaLibMgMatrixDesc_t desc);

    /**
     * \brief Allocates resources related to the shared memory 2D device grid.
     * \param[out] grid the opaque data strcuture that holds the grid
     * \param[in] numRowDevices number of devices in the row
     * \param[in] numColDevices number of devices in the column
     * \param[in] deviceId This array of size numRowDevices * numColDevices stores the
     *            device-ids of the 2D grid; each entry must correspond to a valid gpu.
     * \param[in] mapping whether the 2D device grid is linearized in row/column major
     * \returns the status code
     */
    cudaLibMgStatus_t cudaLibMgCreateDeviceGrid(cudaLibMgGrid_t * grid, 
            int32_t numRowDevices, int32_t numColDevices, const int32_t deviceIds[], cudaLibMgGridMapping_t mapping);

    /**
     * \brief Releases the allocated resources related to the grid.
     * \param[in] grid the opaque data strcuture that holds the grid
     * \returns the status code
     */
    cudaLibMgStatus_t cudaLibMgDestroyGrid(cudaLibMgGrid_t grid);

    /**
     * \brief Calculates the local matrix dimensions for each device.
     * \param[in] desc the opaque data strcuture that holds the matrix descriptor
     * \param[out] numRows[i] denotes the number of rows of the local matrix stored on
     *             the ith device associated to the grid (see deviceId of cudaLidMgCreateDeviceGrid).
     * \param[out] numCols[i] denotes the number of columns of the local matrix stored on
     *             the ith device associated to the grid.
     * \returns the status code
     */
    cudaLibMgStatus_t cudaLibMgGetLocalMatrixDimensions(const cudaLibMgMatrixDesc_t desc,
            int64_t * numRows, int64_t *numCols);
};

#endif /* define CUDALIBMG_H */
