/*********************************************************
 * Wrapper to create cross-product pairs of benchmarks
 *                Created by Serina Tan 
 *                     Apr 5, 2019 
 * *******************************************************
 * ******************************************************/



#include <stdio.h>
#include <iostream>
#include <functional>
#include <string>

#include "cuda_runtime.h"

#include "interface.h"


int main(int argc, char** argv) {
  if (argc < 11 || argv[0] == "-h") {
    std::cout << "Usage: ";
    std::cout << "./PAIR <--APP1> <APP1 args> <--APP2> <APP2 args>";
  } 

  // passing inputs
  std::string A_str, B_str;
  int A_idx, B_idx;
  int idx = 1;
  bool done_A = false;

  while (idx < argc) {
    if (strlen(argv[idx]) >= 3 && ( strncmp("--",argv[idx],2) == 0 )) {
      if (!done_A) {
        A_str = std::string(argv[idx]);
        A_idx = idx + 1;

        done_A = true;
      } else {
        B_str = std::string(argv[idx]);
        B_idx = idx + 1;
      }
    }

    idx++;
  }

  const int argc_A = B_idx-A_idx-1;
  const int argc_B = argc-B_idx;

  // create two different cuda streams
  cudaStream_t stream_A, stream_B;

  cudaStreamCreate(&stream_A);
  cudaStreamCreate(&stream_B);

  // grab kernel launch and exit function calls from benchmark A and B
  std::function<int(const int, cudaStream_t &)> kernel_A, kernel_B;
  std::function<void(void)> exit_A, exit_B;

  // FIXME: temp pointing to sgemm and spmv
  main_sgemm(argc_A, &(argv[A_idx]), kernel_A, exit_A);

  main_spmv(argc_B, &(argv[B_idx]), kernel_B, exit_B);


  // run the kernels
  kernel_A(1, stream_A);
  std::cout << "done A" << std::endl;
  kernel_B(1, stream_B);
  std::cout << "done B" << std::endl;

  exit_A();
  exit_B();


  return 0;
}
