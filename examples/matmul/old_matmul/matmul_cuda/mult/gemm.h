#ifndef GEMM_HPP
#define GEMM_HPP

#include <stdio.h>

//void gemm(int m, int n, int k, const float *A, const float *B, float *C);
void internal_gemm(int m, int n, int k, const float *A, const float *B, float *C, int device);
void make_handle(int device);

#endif //GEMM_HPP