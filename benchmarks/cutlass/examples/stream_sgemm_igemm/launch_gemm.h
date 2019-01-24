#ifndef LAUNCH_GEMM_H
#define LAUNCH_GEMM_H

#include "cuda.h"

// Too lazy to make these polymorphic
struct float_mm_info {
    int M;
    int N;
    int K;
    int lda;
    int ldb;
    int ldc;
    int nitems_A;
    int nitems_B;
    int nitems_C;

    int num_matrices;

    float alpha; 
    float beta;

    float* A;
    float* B;
    float* C_cutlass;
    float* C_reference;

};

struct int_mm_info {
    int M;
    int N;
    int K;
    int lda;
    int ldb;
    int ldc;
    int nitems_A;
    int nitems_B;
    int nitems_C;

    int8_t alpha;
    int8_t beta;

    int8_t* A;
    int8_t* B;
    int* C_cutlass;
    int* C_reference;

};



// SGEMM stuff
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  cudaStream_t& stream);

cudaError_t SetupSgemm(float_mm_info sgemm_info);

cudaError_t ValidateSgemm(float_mm_info sgemm_info, int niter);


// IGEMM stuff



#endif


