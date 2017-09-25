#include <stdint.h>
#include <inttypes.h>
#include <cublas_v2.h>
#include <stdio.h>

extern "C"
{
//__global__ void kComputeMatMult(const int K,const int M,const int N, const float* A, const float* B, long long* C);
__global__ void kComputeDistances(const int K,const int M,const int N, const float* A, const float* B, float* C);
}
