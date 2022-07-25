/**
 * @file
 * @brief This file contains all of cublasMg's public function declarations.
 */
#pragma once

#define CUBLASMG_MAJOR 0 //!< cuBLASMg major version.
#define CUBLASMG_MINOR 1 //!< cuBLASMg minor version.
#define CUBLASMG_PATCH 1 //!< cuBLASMg patch version.

#include <cublas_v2.h>
#include <cudalibmg.h>

#include <cublasmg/types.h>

/**
 * \mainpage cublasMg: Multi-GPU, out-of-core distributed BLAS
 *
 * \section intro Introduction
 * This multi-GPU library enables users to compute BLAS-like operations on distributed
 * matrices.
 *
 * \section api API Reference
 * For details on the API please see \ref cublasMg.h.
 */


extern "C"
{
    /**
     * Multi-GPU, mixed precision GEMM of the form D = alpha * A * B + beta * C
     *
     * This function enables users to compute a matrix-matrix multiplication of the form D = alpha * A * B + beta * C
     * with the matrices A, B, C, D being distributed across multiple devices.
     *
     * \param[in] handle Holds the cublasMg library context.
     * \param[in] transA operation op(A) that is non- or (conj.) transpose.
     * \param[in] transB operation op(B) that is non- or (conj.) transpose.
     * \param[in] alpha Scaling factor for A*B; of same type as computeType.
     * \param[in] descA Dense, distributed matrix descriptor for A. It must be ensured
     * that all devices that own data of A must also be specified via cublasMgDeviceSelect().
     * \param[in] A This array stores the pointers to the distributed sub-matrices of A.
     * For instance, let the underlying process grid consist of x devices, then
     * for all 0 <= i < x: A[i] corresponds to a buffer that resides on the device deviceId[i];
     * A[i] denotes a local 2D sub-matrix of A roughly of size m/numProcessesM * n/numProcessesN.
     * The individual sub-matrices may reside on various devices (including CPU),
     * as long as these buffers belong to the unified virtual address space (UVM).
     * \param[in] llda This array stores the leading-dimensions (in elements) of the sub-matrices A[i].
     * \param[in] descB Dense, distributed matrix descriptor for B (see descA).
     * \param[in] B This array stores the pointers to the distributed sub-matrices of B.
     * \param[in] lldb This array stores the leading-dimensions (in elements) of the sub-matrices B[i].
     * \param[in] beta Scaling factor for C; of same type as computeType.
     * \param[in] descC Dense, distributed matrix descriptor for C.
     * \param[in] C This array stores the pointers to the distributed sub-matrices of C.
     * \param[in] lldc This array stores the leading-dimensions (in elements) of the sub-matrices c[i].
     * \param[in] descD Descriptor for output matrix (must be identical to descC for now).
     * \param[in] D This array stores the pointers to the distributed sub-matrices of D (must be identical to C for now).
     * \param[in] lldd leading dimensions of the output matrix (must be identical to lldc for now).
     * \param[in] computeType datatype used during the calculation (see cublasGemmEx()).
     * \param[out] workspace Device-side scratchpad memory for each GPU (see cublasMgGemmWorkspace()).
     * \param[in] lwork Size of the workspace on each GPU in bytes (see cublasMgGemmWorkspace()).
     * \param[in] streams Stream i must corrspond to GPU with the id deviceIds[i] that was provided to
     *            cublasMgDeviceSelect(). While cuBLASMg uses internal streams, it will
     *            look to the user as if all tasks that are scheduled to GPU i will be be
     *            part of stream i (i.e., the user can start issuing new tasks to stream i
     *            immidiately after this call returns).
     * \return CUBLAS_STATUS_SUCCESS on success, otherwise an error status is reported.
     * \behavior non-blocking
     */
    cublasStatus_t cublasMgGemm( const cublasMgHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
            const void *alpha, const cudaLibMgMatrixDesc_t descA, void const* const A[], const int64_t llda[],
                               const cudaLibMgMatrixDesc_t descB, void const* const B[], const int64_t lldb[],
            const void *beta,  const cudaLibMgMatrixDesc_t descC, void const* const C[], const int64_t lldc[],
                               const cudaLibMgMatrixDesc_t descD, void      * const D[], const int64_t lldd[],
            cudaDataType_t computeType, void * const workspace[], const size_t lwork[], const cudaStream_t streams[]);


    /**
     * This function calculates the required workspace for each GPU for a corresponding call to cublasMgGemm().
     *
     * \param[out] lwork Size of the workspace on each GPU in bytes.
     */
    cublasStatus_t cublasMgGemmWorkspace( const cublasMgHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
            const void *alpha, const cudaLibMgMatrixDesc_t descA, void const* const A[], const int64_t llda[],
                               const cudaLibMgMatrixDesc_t descB, void const* const B[], const int64_t lldb[],
            const void *beta,  const cudaLibMgMatrixDesc_t descC, void const* const C[], const int64_t lldc[],
                               const cudaLibMgMatrixDesc_t descD, void const* const D[], const int64_t lldd[],
            cudaDataType_t computeType, void const* const workspace[], size_t* lwork );


    /**
     * Allocates the resources necessary to store cublasMg' context
     *
     * \param[out] handle Holds the cublasMg library context.
     */
    cublasStatus_t cublasMgCreate(cublasMgHandle_t *handle);

    /**
     * Frees cublasMg' context
     *
     * \param[in,out] handle Holds the cublasMg library context.
     */
    cublasStatus_t cublasMgDestroy(cublasMgHandle_t handle);

    /**
     * This function allows the user to provide the number of GPU devices and their
     * respective Ids that will participate to the subsequent cublasMg API Math function
     * calls. This function will create a cuBLAS context for every GPU provided in that
     * list. Currently the device configuration is static and cannot be changed between
     * Math function calls. In that regard, this function should be called only once after
     * cublasMgCreate. To be able to run multiple configurations, multiple cublasMg API
     * contexts should be created.
     * IMPORTANT: This function will enable peer-to-peer accesses among these devices, if
     * possible.
     *
     * \param[in,out] handle Holds the cublasMg library context.
     * \param[in] nbDevices Number of entries in deviceIds array.
     * \param[in] deviceId Array of nbDevices that holds the devices IDs of the to-be-used
     * devices.
     */
    cublasStatus_t cublasMgDeviceSelect(cublasMgHandle_t handle, int nbDevices, const int deviceIds[]);

    /**
     * Returns the number of devices used by the provided handle.
     */
    cublasStatus_t cublasMgDeviceCount(const cublasMgHandle_t handle, int *nbDevices);

    /**
     * Enables (or disables) tensor core support.
     *
     * \param[in,out] handle Holds the cublasMg library context.
     */
    cublasStatus_t cublasMgUseTensorCores(cublasMgHandle_t handle, bool useTensorCores );

    /**
     * \brief Returns Version number of the cuBLASMg library.
     */
    size_t cublasMgGetVersion();

    /**
     * \brief Returns version number of the CUDA runtime that cuBLASMg was compiled against.
     */
    size_t cublasMgGetCudartVersion();
}
