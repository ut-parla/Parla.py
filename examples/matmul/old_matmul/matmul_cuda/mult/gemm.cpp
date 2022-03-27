#include "gemm.h"

void gemm(int m, int n, int k, const float *A, const float *B, float *C, int device)
{
    internal_gemm(m, n, k, A, B, C, device);
}
