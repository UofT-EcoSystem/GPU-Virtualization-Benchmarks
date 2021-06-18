//
// Created by Serina Tan on 2021-06-17.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "binomialOptions_common.h"
#include "realtype.h"


extern void call_binomial_kernel(cudaStream_t stream);
////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for binomial tree results validation
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCall(
    real &callResult,
    TOptionData optionData
);

////////////////////////////////////////////////////////////////////////////////
// Process single option on CPU
// Note that CPU code is for correctness testing only and not for benchmarking.
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsCPU(
    real &callResult,
    TOptionData optionData
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsGPU(
    real *callValue,
    TOptionData  *optionData,
    int optN,
    int uid,
    cudaStream_t & stream
);

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
real randData(real low, real high)
{
  real t = (real)rand() / (real)RAND_MAX;
  return ((real)1.0 - t) * low + t * high;
}



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main_binomial(int argc, char** argv, int uid, cudaStream_t & stream)
{
  printf("[%s] - Starting...\n", argv[0]);

  int devID = findCudaDevice(argc, (const char **)argv);

  const int OPT_N = MAX_OPTIONS;

  TOptionData optionData[MAX_OPTIONS];
  real
      callValueBS[MAX_OPTIONS],
      callValueGPU[MAX_OPTIONS],
      callValueCPU[MAX_OPTIONS];

  real
      sumDelta, sumRef, gpuTime, errorVal;

  StopWatchInterface *hTimer = NULL;
  int i;

  sdkCreateTimer(&hTimer);

  printf("Generating input data...\n");
  //Generate options set
  srand(123);

  for (i = 0; i < OPT_N; i++)
  {
    optionData[i].S = randData(5.0f, 30.0f);
    optionData[i].X = randData(1.0f, 100.0f);
    optionData[i].T = randData(0.25f, 10.0f);
    optionData[i].R = 0.06f;
    optionData[i].V = 0.10f;
    BlackScholesCall(callValueBS[i], optionData[i]);
  }

  printf("Running GPU binomial tree...\n");
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  binomialOptionsGPU(callValueGPU, optionData, OPT_N, uid, stream);

  checkCudaErrors(cudaStreamSynchronize(stream));
  return 0;
}

int main(int argc, char** argv) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  main_binomial(argc, argv, 0, stream);
  call_binomial_kernel(stream);
}


