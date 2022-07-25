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
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cudalibmg.h>
#include <cublasMg.h>

#define HANDLE_CUDA_ERROR(x) { const auto err = x; if(err != cudaSuccess ){ printf("Error: %s in %s Line: %d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(-1); } }
#define HANDLE_CUBLAS_ERROR(x) { const auto err = x; if (err != CUBLAS_STATUS_SUCCESS){ printf("Error: in %s Line: %d\n", __FILE__, __LINE__); exit(-1); } }
#define HANDLE_CUBLASMG_ERROR(x) { const auto err = x; if (err != CUBLAS_STATUS_SUCCESS){ printf("Error: in %s Line: %d\n", __FILE__, __LINE__); exit(-1); } }
#define HANDLE_CUDALIBMG_ERROR(x) { const auto err = x; if (err != CUDALIBMG_STATUS_SUCCESS){ printf("Error: in %s Line: %d\n", __FILE__, __LINE__); exit(-1); } }

template<typename floatType>
void allocateBuffers(void** distBuffers, int64_t* lld, cudaLibMgGrid_t grid, int numDevices, 
        const int* deviceIdsGrid, cudaLibMgMatrixDesc_t desc)
{
    int currentDeviceId;
    HANDLE_CUDA_ERROR(cudaGetDevice(&currentDeviceId));
    int64_t numRows[numDevices];
    int64_t numCols[numDevices];
    HANDLE_CUDALIBMG_ERROR(cudaLibMgGetLocalMatrixDimensions(desc, numRows, numCols));
    size_t elementSize = sizeof(floatType); 
    for (int i=0; i < numDevices; ++i)
    {
        lld[i] = numRows[i];
        int id = deviceIdsGrid[i];
        size_t size = ((size_t)numRows[i]) * ((size_t)numCols[i]) * elementSize;
        if( id != -1 )
        {
            HANDLE_CUDA_ERROR(cudaSetDevice(id));
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&distBuffers[i], size));
        }
        else
        {
            HANDLE_CUDA_ERROR(cudaHostAlloc((void**)&distBuffers[i], size, cudaHostAllocWriteCombined));
        }
    }
    HANDLE_CUDA_ERROR(cudaSetDevice(currentDeviceId));
}

template<typename T=float>
void multiDeviceGemm()
{
    cudaDataType_t datatype;
    if (std::is_same<T, float>::value)
    {
        datatype = CUDA_R_32F;
    }
    else if (std::is_same<T, double>::value)
    {
        datatype = CUDA_R_64F;
    }
    else
    {
        std::cout << "datatype not supported in this example\n";
        return;
    }

    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_T;
    const T alpha = 1.0;
    const T beta = 0.0;
    int64_t m = 32000;
    int64_t n = 32000;
    int64_t k = 32000;
    const int numDevicesM = 2;
    const int numDevicesN = 1;
    const int numDevices = numDevicesM * numDevicesN;
    int numGpus = 0;
    int deviceIds[numDevices];
    cudaStream_t streams[numDevices];

    cudaEvent_t start, stop;
    HANDLE_CUDA_ERROR(cudaSetDevice(0));
    HANDLE_CUDA_ERROR(cudaEventCreate(&start));
    HANDLE_CUDA_ERROR(cudaEventCreate(&stop));


    /* Create cublasMg handle */
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numGpus));
    for (int i = 0; i < numDevices; i ++)
    {
        deviceIds[i] = i % numGpus;
        HANDLE_CUDA_ERROR(cudaSetDevice(i));
        HANDLE_CUDA_ERROR(cudaStreamCreate(streams + i));
    }
    HANDLE_CUDA_ERROR(cudaSetDevice(0));
    cublasMgHandle_t handleMg;
    HANDLE_CUBLASMG_ERROR(cublasMgCreate(&handleMg));
    HANDLE_CUBLASMG_ERROR(cublasMgDeviceSelect(handleMg, numDevices, deviceIds));

    /* Create a 2D device grid and matrix descriptors */
    cudaLibMgGrid_t grid;
    cudaLibMgMatrixDesc_t descA;
    cudaLibMgMatrixDesc_t descB;
    cudaLibMgMatrixDesc_t descC;
    HANDLE_CUDALIBMG_ERROR(cudaLibMgCreateDeviceGrid(&grid, numDevicesM, numDevicesN, deviceIds, 
                CUDALIBMG_GRID_MAPPING_ROW_MAJOR));
    HANDLE_CUDALIBMG_ERROR(cudaLibMgCreateMatrixDesc(&descA, 
            (opA == CUBLAS_OP_N) ? m : k, (opA == CUBLAS_OP_N) ? k : m, 2048, 2048, datatype, grid));
    HANDLE_CUDALIBMG_ERROR(cudaLibMgCreateMatrixDesc(&descB, 
            (opB == CUBLAS_OP_N) ? k : n, (opB == CUBLAS_OP_N) ? n : k, 2048, 2048, datatype, grid));
    HANDLE_CUDALIBMG_ERROR(cudaLibMgCreateMatrixDesc(&descC, 
            m, n, 2048, 2048, datatype, grid));
    /* Allocate device buffers */
    void* distA[numDevices];
    void* distB[numDevices];
    void* distC[numDevices];
    int64_t llda[numDevices];
    int64_t lldb[numDevices];
    int64_t lldc[numDevices];
    /* See cudaLibMg redistribute example for redistributing a host matrix to devices. */
    allocateBuffers<T>(distA, llda, grid, numDevices, deviceIds, descA);
    allocateBuffers<T>(distB, lldb, grid, numDevices, deviceIds, descB);
    allocateBuffers<T>(distC, lldc, grid, numDevices, deviceIds, descC);

    /* Querry required lwork. */
    void * workspace[numDevices];
    size_t lwork[numDevices];
    HANDLE_CUBLASMG_ERROR(cublasMgGemmWorkspace( handleMg, opA, opB,
            &alpha, descA, distA, llda,
                    descB, distB, lldb,
            &beta,  descC, distC, lldc,
                    descC, distC, lldc,
            datatype, workspace, lwork));
    /* Allocate worspace */
    for(int i=0; i < numDevices; ++i)
    {
        HANDLE_CUDA_ERROR(cudaSetDevice(deviceIds[i]));
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&workspace[i], lwork[i]));
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    }
    HANDLE_CUDA_ERROR(cudaSetDevice(0));

    int nRuns = 3;
    /***************************************
     * time cublasMg
     ***************************************/
    double timecublasmg = 1e100;
    for (int i=0; i < nRuns; ++i)
    {
        // Synchronize all devices
        for(int j=0; j < numDevices; ++j)
        {
            HANDLE_CUDA_ERROR(cudaSetDevice(deviceIds[j]));
            HANDLE_CUDA_ERROR(cudaStreamSynchronize(streams[j]));
        }
        HANDLE_CUDA_ERROR(cudaSetDevice(0));

        HANDLE_CUDA_ERROR(cudaEventRecord(start));
        HANDLE_CUBLASMG_ERROR(cublasMgGemm( handleMg, opA, opB,
                &alpha, descA, distA, llda,
                descB, distB, lldb,
                &beta,  descC, distC, lldc,
                        descC, distC, lldc,
                datatype, workspace, lwork, streams));

        // Synchronize all devices
        for(int j=0; j < numDevices; ++j)
        {
            HANDLE_CUDA_ERROR(cudaSetDevice(deviceIds[j]));
            HANDLE_CUDA_ERROR(cudaStreamSynchronize(streams[j]));
        }
        HANDLE_CUDA_ERROR(cudaSetDevice(0));

        float elapsed = 0;
        HANDLE_CUDA_ERROR(cudaEventRecord(stop));
        HANDLE_CUDA_ERROR(cudaEventSynchronize(stop));
        HANDLE_CUDA_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
        double time = (double)elapsed / 1000.;
        timecublasmg = std::min(time, timecublasmg);
    }

    const double gflops = 2. * m * n * k / 1e9;
    printf("%ld %ld %ld %s %s: %.2f GFLOPs/s\n", m, n, k,
            (opA == CUBLAS_OP_N)? "N" : "T", (opB == CUBLAS_OP_N)? "N" : "T", timecublasmg);

    /* Clean up */
    HANDLE_CUBLASMG_ERROR(cublasMgDestroy(handleMg));
    for (int id=0; id < numDevices; ++id)
    {
        if (lwork[id] > 0)
        {
            HANDLE_CUDA_ERROR(cudaFree(workspace[id]));
        }
        HANDLE_CUDA_ERROR(cudaStreamDestroy(streams[id]));
    }
    HANDLE_CUDALIBMG_ERROR(cudaLibMgDestroyMatrixDesc(descA));
    HANDLE_CUDALIBMG_ERROR(cudaLibMgDestroyMatrixDesc(descB));
    HANDLE_CUDALIBMG_ERROR(cudaLibMgDestroyMatrixDesc(descC));
    HANDLE_CUDALIBMG_ERROR(cudaLibMgDestroyGrid(grid));
}

int main(int argc, char* argv[])
{
    multiDeviceGemm<float>();
    //multiDeviceGemm<double>();
    return 0;
}
