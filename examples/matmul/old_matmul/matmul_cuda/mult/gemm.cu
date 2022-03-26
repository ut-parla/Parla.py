#include "cuda_profiler_api.h"
#include "singleton.h"
#include "gemm.h"

void make_handle(int device){
  cudaSetDevice(device);
  auto const& handle = knnHandle_t::instance();
  cudaDeviceSynchronize();
}


void internal_gemm(int m, int n, int k, const float* A, const float* B, float* C, int device) {
  cudaSetDevice(device);
  auto const& handle = knnHandle_t::instance();
  float alpha = 1.0;
  float beta = 0.;
  cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N, 
        m, n, k, &alpha,
        A, m,
        B, k, &beta,
        C, m);
  cudaDeviceSynchronize();
}
