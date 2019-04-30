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

void invoke(std::string kernel_str,
            int argc,
            char** argv,
            std::function<int(const int, cudaStream_t &)>& kernel,
            std::function<void(void)>& cleanup)
{
  // select the right benchmark symbol
  if (kernel_str.compare("+spmv") == 0) {
    std::cout << "main: spmv" << std::endl;
    main_spmv(argc, argv, kernel, cleanup);
  } else if (kernel_str.compare("+sgemm") == 0) {
    std::cout << "main: sgemm" << std::endl;
    main_sgemm(argc, argv, kernel, cleanup);
  } else if (kernel_str.compare("+cutcp") == 0) {
    std::cout << "main: cutcp" << std::endl;
    main_cutcp(argc, argv, kernel, cleanup);
  } else if (kernel_str.compare("+mri-q") == 0) {
    std::cout << "main: mri-q" << std::endl;
    main_mriq(argc, argv, kernel, cleanup);
  } else if (kernel_str.compare("+tpacf") == 0) {
    std::cout << "main: tpacf" << std::endl;
    main_tpacf(argc, argv, kernel, cleanup);
  } else if (kernel_str.compare("+lbm") == 0) {
    std::cout << "main: lbm" << std::endl;
    main_lbm(argc, argv, kernel, cleanup);
  }
  else {
    std::cout << "Warning: No matching kernels!" << std::endl;
  }

}


int main(int argc, char** argv) {
  if (argc < 8 || argv[1] == "-h") {
    std::cout << "Usage: ";
    std::cout << "./PAIR [1|2|b] <--APP1> <APP1 args> <--APP2> <APP2 args>" << std::endl;
    abort();
  } 

  // passing inputs
  std::string A_str, B_str;
  int A_idx, B_idx;
  int idx = 2;
  bool done_A = false;

  while (idx < argc) {
    if (strlen(argv[idx]) >= 3 && ( strncmp("+",argv[idx],1) == 0 )) {
      if (!done_A) {
        A_str = std::string(argv[idx]);
        A_idx = idx;

        done_A = true;
      } else {
        B_str = std::string(argv[idx]);
        B_idx = idx;
      }
    }

    idx++;
  }

  const int argc_A = B_idx-A_idx;
  const int argc_B = argc-B_idx;



  // run the kernels
  if (strncmp("1",argv[1],1) == 0) {
    cudaStream_t stream_A;
    cudaStreamCreate(&stream_A);

    // grab kernel launch and exit function calls from benchmark A and B
    std::function<int(const int, cudaStream_t &)> kernel_A;
    std::function<void(void)> cleanup_A;

    invoke(A_str, argc_A, &(argv[A_idx]), kernel_A, cleanup_A);

    kernel_A(1, stream_A);

    cleanup_A();

  } else if (strncmp("2", argv[1], 1) == 0) {
    cudaStream_t stream_B;
    cudaStreamCreate(&stream_B);

    // grab kernel launch and exit function calls from benchmark A and B
    std::function<int(const int, cudaStream_t &)> kernel_B;
    std::function<void(void)> cleanup_B;

    invoke(B_str, argc_B, &(argv[B_idx]), kernel_B, cleanup_B);

    kernel_B(1, stream_B);

    cleanup_B();

  } else {
    // run both
    // create two different cuda streams
    cudaStream_t stream_A, stream_B;

    cudaStreamCreate(&stream_A);
    cudaStreamCreate(&stream_B);

    // grab kernel launch and exit function calls from benchmark A and B
    std::function<int(const int, cudaStream_t &)> kernel_A, kernel_B;
    std::function<void(void)> cleanup_A, cleanup_B;

    invoke(A_str, argc_A, &(argv[A_idx]), kernel_A, cleanup_A);
    invoke(B_str, argc_B, &(argv[B_idx]), kernel_B, cleanup_B);

    for (int i = 0; i < 5; i++) {
      kernel_A(1, stream_A);
      kernel_B(1, stream_B);
    }

    cleanup_A();
    cleanup_B();


  }

  return 0;
}
