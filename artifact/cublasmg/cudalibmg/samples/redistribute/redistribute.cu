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
#include <cassert>
#include <algorithm>
#include <cudalibmg.h>

#define HANDLE_CUDA_ERROR(x) \
{ \
    const auto err = x; \
    if (err != cudaSuccess) \
    { \
        printf("Error: %s in %s Line: %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(-1); \
    } \
}

#define HANDLE_CUDALIBMG_ERROR(x) \
{ \
    const auto err = x; \
    if (err != CUDALIBMG_STATUS_SUCCESS) \
    { \
        printf("Error: in %s Line: %d\n", __FILE__, __LINE__); \
        exit(-1); \
    } \
}

void redistributeFromHost(int64_t numRows, int64_t numCols, int64_t rowBlockSize, int64_t colBlockSize, size_t elementSize,
        int32_t numRowDevices, int32_t numColDevices, int32_t deviceId[], 
        void * const * A_d, const int64_t * llda, const void * A_h, const int64_t lda) 
{
    for (int64_t j = 0; j < numCols; j += colBlockSize)
    {
        for (int64_t i = 0; i < numRows; i += rowBlockSize)
        {
            /* Acquire local matrix information */
            int64_t mb = std::min(rowBlockSize, numRows - i);
            int64_t nb = std::min(colBlockSize, numCols - j);
            int64_t ib = i / rowBlockSize;
            int64_t jb = j / colBlockSize;
            int64_t rowOffset = (ib / numRowDevices) * rowBlockSize;
            int64_t colOffset = (jb / numColDevices) * colBlockSize;
            int32_t i_grid = ib % numRowDevices;
            int32_t j_grid = jb % numColDevices;
            int32_t owner = i_grid * numColDevices + j_grid;
            /* Prepare inputs for cudaMemcpy2D */
            const char * src = static_cast<const char*>(A_h) + (j * lda + i) * elementSize;
            char * dst = static_cast<char*>(A_d[owner]) + (colOffset * llda[owner] + rowOffset) * elementSize; 
            size_t spitch = lda * elementSize;
            size_t dpitch = llda[owner] * elementSize;
            size_t width = mb * elementSize;
            size_t height = nb;
            HANDLE_CUDA_ERROR(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDefault));
        }
    }
}

/**
 * \brief This example illustrates how to redistribute a host matrix A according to the
 *        3-by-2 device grid and the matrix descriptor.
 */ 
int main(int argc, char * argv[])
{
    const int32_t numRowDevices = 3;
    const int32_t numColDevices = 2;
    const int32_t numDevices = numRowDevices * numColDevices;
    int32_t deviceId[numDevices] = {0};
    cudaLibMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_ROW_MAJOR;
    cudaLibMgGrid_t grid;
    HANDLE_CUDALIBMG_ERROR(cudaLibMgCreateDeviceGrid(&grid, numRowDevices, numColDevices, deviceId, mapping));
    const int64_t numRows = 7;
    const int64_t numCols = 9;
    const int64_t rowBlockSize = 3;
    const int64_t colBlockSize = 2;
    cudaDataType_t dataType = CUDA_R_32F;
    cudaLibMgMatrixDesc_t descA;
    HANDLE_CUDALIBMG_ERROR(cudaLibMgCreateMatrixDesc(&descA, 
                numRows, numCols, rowBlockSize, colBlockSize, dataType, grid));
    /* 
     * We continue to use this 7-by-9 matrix and a 3-by-2 grid using a 2D block-cyclic matrix distribution.
     * We store A in a column-major format.
     */ 
    float A[7 * 9] = {0, 0, 0, 2, 2, 2, 4, 
                      0, 0, 0, 2, 2, 2, 4, 
                      1, 1, 1, 3, 3, 3, 5, 
                      1, 1, 1, 3, 3, 3, 5, 
                      0, 0, 0, 2, 2, 2, 4, 
                      0, 0, 0, 2, 2, 2, 4, 
                      1, 1, 1, 3, 3, 3, 5, 
                      1, 1, 1, 3, 3, 3, 5, 
                      0, 0, 0, 2, 2, 2, 4}; 
    /*
     * Observe that the matrix above shall be distributed into numDecices (=6) local submatrices. 
     * We need to create corresponding device memory to hold each local submatrix separately. We use 
     * cudaLibMgGetLocalMatrixDimensions() to compute the dimension of each local submatrix.
     */ 
    void * A_d[numDevices];
    float * A_h[numDevices];
    int64_t rows[numDevices], columns[numDevices];
    HANDLE_CUDALIBMG_ERROR(cudaLibMgGetLocalMatrixDimensions(descA, rows, columns));
    for (int32_t rank = 0; rank < numDevices; rank ++)
    {
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&A_d[rank], sizeof(float) * rows[rank] * columns[rank]));
        A_h[rank] = new float[rows[rank] * columns[rank]];
    }
    /* Redistribute A to 2D block cyclic. */
    redistributeFromHost(numRows, numCols, rowBlockSize, colBlockSize, sizeof(float),
        numRowDevices, numColDevices, deviceId, A_d, rows, A, numRows); 
    /* Validate the results. */
    for (int32_t rank = 0; rank < numDevices; rank ++)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy(A_h[rank], A_d[rank], sizeof(float) * rows[rank] * columns[rank], cudaMemcpyDefault));
        printf("rank%d\n", rank);
        for (int64_t j = 0; j < columns[rank]; j ++)
        {
            for (int64_t i = 0; i < rows[rank]; i ++)
            {
                printf("%f ", A_h[rank][j * rows[rank] + i]);
            }
            printf("\n");
        }
    }
    /* Clean up. */
    for (int32_t rank = 0; rank < numDevices; rank ++)
    {
        HANDLE_CUDA_ERROR(cudaFree(A_d[rank]));
    }
    HANDLE_CUDALIBMG_ERROR(cudaLibMgDestroyMatrixDesc(descA));
    HANDLE_CUDALIBMG_ERROR(cudaLibMgDestroyGrid(grid));
    return 0;
};



