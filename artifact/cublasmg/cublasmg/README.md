# cuBLASMg: Multi-GPU BLAS #

This library provides multi-gpu BLAS functionality for CUDA-enabled devices.

# Key Features

* Out-of-core, multi-gpu GEMMs
* Over-subscribe GPU memory
* Mixed precision support
* Matrices can be distributed in a 2D-block-cyclic fashon across multiple devices
* High performance

# Requirements

* CUDA toolkit 10.1

# Getting Started

Please have a look at the provided examples in ./samples/

# Documentation

Please see include/cublasMg.h for the API documentation.
You could also generate the doxygen documentation yourself via:

    doxygen

# C-Interface

cuBLASMg offers a similar C interface as cublasGemmEx (see [cublas documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublas-gemmEx) for details).

    cublasStatus_t cublasMgGemm( const cublasMgHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
            const void *alpha, const cudaLibMgMatrixDesc_t descA, void const* const A[], const int64_t llda[],
                               const cudaLibMgMatrixDesc_t descB, void const* const B[], const int64_t lldb[],
            const void *beta,  const cudaLibMgMatrixDesc_t descC, void const* const C[], const int64_t lldc[],
                               const cudaLibMgMatrixDesc_t descD, void      * const D[], const int64_t lldd[],
            cudaDataType_t computeType, void * const workspace[], const size_t lwork[], const cudaStream_t streams[]);



# Citation
