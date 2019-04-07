/*********************************************************
 * Wrapper to create cross-product pairs of benchmarks
 *                Created by Serina Tan 
 *                     Apr 5, 2019 
 * *******************************************************
 * ******************************************************/



#include <stdio.h>
#include <iostream>
#include <functional>

#include "cuda_runtime.h"

#include "interface.h"


const int sgemm_argc = 4;
char sgemm_argv[sgemm_argc][200] = {"-i", 
                        "sgemm/medium/input/matrix1.txt, \
                          sgemm/medium/input/matrix2t.txt, \
                          sgemm/medium/input/matrix2t.txt",
                        "-o",
                        "sgemm/run/medium/matrix3.txt"};

const int spmv_argc = 4;
char spmv_argv[spmv_argc][200] {"-i",
                    "spmv/large/input/Dubcova3.mtx.bin, \
                      spmv/large/input/vector.bin",
                    "-o",
                    "spmv/run/large/Dubcova3.mtx.out"};

int main() {

  // create two different cuda streams
  cudaStream_t stream_A, stream_B;

  cudaStreamCreate(&stream_A);
  cudaStreamCreate(&stream_B);

  // grab kernel launch and exit function calls from benchmark A and B
  std::function<void(const int, cudaStream_t &)> kernel_A, kernel_B;
  std::function<void(void)> exit_A, exit_B;

  // FIXME: temp pointing to sgemm and spmv
  main_sgemm(sgemm_argc, (char**)sgemm_argv, kernel_A, exit_A);
  main_spmv(spmv_argc, (char**)spmv_argv, kernel_B, exit_B);

  // run the kernels
  kernel_A(1, stream_A);
  kernel_B(1, stream_B);

  exit_A();
  exit_B();


  return 0;
}
