/* 
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 * 
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <cudalibmg.h>

#define HANDLE_CUDALIBMG_ERROR(x) \
{ \
    const auto err = x; \
    if (err != CUDALIBMG_STATUS_SUCCESS) \
    { \
        printf("Error: in %s Line: %d\n", __FILE__, __LINE__); \
        exit(-1); \
    } \
}

/**
 * \brief This example illustrates how to create and destroy a matrix descriptor
 *        using a 3-by-2 device grid. 
 */ 
int main(int argc, char * argv[])
{
    /* Create a 3-by-2 device grid. */
    const int32_t numRowDevices = 3;
    const int32_t numColDevices = 2;
    const int32_t numDevices = numRowDevices * numColDevices;
    /* 
     * cudaLibMg allow duplicated deviceIds. this example only uses GPU 0. As a result, the matrix created based
     * on this 3-by-2 grid will be distributed as 6 piceses but all on the same device. If you have more devices
     * available, change the device list below.
     */
    int32_t deviceId[numDevices] = {0};
    /*
     * Assign the 1D deviceId to this 3-by-2 grid using row-major:
     * 
     * grid = [ deviceId[0], deviceId[1];
     *        [ deviceId[2], deviceId[3];
     *        [ deviceId[4], deviceId[5]; ];
     */ 
    cudaLibMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_ROW_MAJOR;
    cudaLibMgGrid_t grid;
    HANDLE_CUDALIBMG_ERROR(cudaLibMgCreateDeviceGrid(&grid, numRowDevices, numColDevices, deviceId, mapping));
    /*
     * We use the rank to illustrate how this 7-by-9 matrix is distributed on a 3-by-2 grid using 2D block-cyclic
     * matrix distribution. This distribution is designed for load-balancing and preserve the data locality in mind.
     * We can observe a 3-by-2 matrix block is owned entirely by a rank in a cyclic fashion
     * in both row and column directions. 
     *
     * float A[7 * 9] = {0, 0, 0,    2, 2, 2,    4, 
     *                   0, 0, 0,    2, 2, 2,    4, 
     *
     *                   1, 1, 1,    3, 3, 3,    5, 
     *                   1, 1, 1,    3, 3, 3,    5, 
     *
     *                   0, 0, 0,    2, 2, 2,    4, 
     *                   0, 0, 0,    2, 2, 2,    4, 
     *
     *                   1, 1, 1,    3, 3, 3,    5, 
     *                   1, 1, 1,    3, 3, 3,    5, 
     *
     *                   0, 0, 0,    2, 2, 2,    4}; 
     */ 
    const int64_t numRows = 7;
    const int64_t numCols = 9;
    const int64_t rowBlockSize = 3;
    const int64_t colBlockSize = 2;
    cudaDataType_t dataType = CUDA_R_32F;
    cudaLibMgMatrixDesc_t descA;
    HANDLE_CUDALIBMG_ERROR(cudaLibMgCreateMatrixDesc(&descA, 
                numRows, numCols, rowBlockSize, colBlockSize, dataType, grid));
    /* Clean up. */
    HANDLE_CUDALIBMG_ERROR(cudaLibMgDestroyMatrixDesc(descA));
    HANDLE_CUDALIBMG_ERROR(cudaLibMgDestroyGrid(grid));
    return 0;
};
