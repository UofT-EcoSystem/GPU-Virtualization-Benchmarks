/*************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm> 
#include <time.h>
#include <inttypes.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <fcntl.h>

#include "launch_gemm.h"

#include "cuda_profiler_api.h"

volatile bool done = false;

void inline print_current_time_with_ms ()
{
    long ms; 
    time_t s;
    struct timespec spec;

    clock_gettime(CLOCK_REALTIME, &spec);

    s = spec.tv_sec;
    ms = round(spec.tv_nsec / 1.0e6);
    if (ms > 999) {
        s++;
        ms = 0;
    }

    printf("Timestamp: %"PRIdMAX": %03ld\n", (intmax_t)s, ms);
}



///////////////////////////////////////////////////////////////////////////////////////////////////

// call a single-precision and integer CUTLASS GEMM kernel.
cudaError_t RunGemm(float_mm_info sgemm_info, int_mm_info igemm_info,
                    int niter) {

  cudaError_t result;

  const int num_streams = 2;
  cudaStream_t streams[num_streams];

  for (int i = 0; i < num_streams; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  cudaProfilerStart();

  int buffer_idx = 0;
  for (int i = 0; i < niter; i++) {
      float* A_adj = &(sgemm_info.A[buffer_idx * sgemm_info.nitems_A]);
      float* B_adj = &(sgemm_info.B[buffer_idx * sgemm_info.nitems_B]);
      float* C_cutlass_adj = &(sgemm_info.C_cutlass[buffer_idx * sgemm_info.nitems_C]);

      result = CutlassSgemmNN(sgemm_info.M, sgemm_info.N, sgemm_info.K, 
                              sgemm_info.alpha, A_adj, sgemm_info.lda, B_adj, 
                              sgemm_info.ldb, sgemm_info.beta, C_cutlass_adj, 
                              sgemm_info.ldc, streams[0]);

      if(buffer_idx >= sgemm_info.num_matrices) buffer_idx = 0;
  }

    cudaProfilerStop();

  if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;

      return result;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////



/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {
  print_current_time_with_ms ();

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[4] = { 128, 128, 128, 10 };

  for (int i = 1; i < argc && i < 5; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 5; i < argc && i < 7; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 5];
  }

  printf("Run CUTLASS matrix multiply A(%d,%d) B(%d,%d) for %d times\n", 
          problem[0], problem[2], problem[2], problem[1], problem[3]);

  float_mm_info sgemm_info;

  sgemm_info.M = problem[0];
  sgemm_info.N = problem[1];
  sgemm_info.K = problem[2];

  sgemm_info.lda = sgemm_info.M;
  sgemm_info.ldb = sgemm_info.K;
  sgemm_info.ldc = sgemm_info.M;

  sgemm_info.nitems_A = sgemm_info.lda * sgemm_info.K;
  sgemm_info.nitems_B = sgemm_info.ldb * sgemm_info.N;
  sgemm_info.nitems_C = sgemm_info.ldc * sgemm_info.N;

  sgemm_info.alpha = scalars[0];
  sgemm_info.beta = scalars[1];

  int_mm_info igemm_info;

  cudaError_t result;
  
  result = SetupSgemm(sgemm_info);

  if (result != cudaSuccess) {
    std::cout << "Failed sgemm input setup." << std::endl;
    return -1;
  }

  result = RunGemm(sgemm_info, igemm_info, problem[3]);

  if (result != cudaSuccess) {
    std::cout << "Failed gemm run." << std::endl;
    return -1;
  }



  //cudaFree(sgemm_info.C_reference);
  cudaFree(sgemm_info.C_cutlass);
  cudaFree(sgemm_info.B);
  cudaFree(sgemm_info.A);


  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

