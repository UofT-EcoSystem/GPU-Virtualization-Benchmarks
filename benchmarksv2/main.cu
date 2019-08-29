/*********************************************************
 * Wrapper to create cross-product pairs of benchmarks
 *                Created by Serina Tan 
 *                     Apr 5, 2019 
 * *******************************************************
 * ******************************************************/


// c++ includes
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <functional>
#include <string>
#include <fstream>
#include <thread>
#include <mutex>
#include <cassert>
#include <vector>
#include <functional>

// cuda include
// #include "cuda_runtime.h"

// user includes
#include "parboil/benchmarks/interface.h"
#include "cutlass/interface.h"

std::vector<bool> done_flags;
std::mutex lock_flag;

bool set_and_check(int uid) {
  // this function is guarded by the mutex
  std::lock_guard<std::mutex> guard(lock_flag);

  if (uid < done_flags.size()) {
    done_flags[uid] = true;
  }

  
  for (auto f : done_flags) {
    if (!f) return false;
  }

  return true;
}

void shared_push() {
  // this function is guarded by the mutex
  std::lock_guard<std::mutex> guard(lock_flag);
  done_flags.push_back(false);
}


void invoke(int uid, std::string kernel_arg)
{
  // split string into argv
  std::vector<std::string> string_argv;
  std::stringstream ss(kernel_arg);
  std::string token;
  while (std::getline(ss, token, ' ')) {
    if (token.length() > 0)
      string_argv.push_back(token);
  }

  // assign each char array
  int argc = string_argv.size();
  char* argv[argc];
  // this vector maintains the original char array pointers
  // cuz the main function will modify the argv
  // this is a sketchy solution
  std::vector<char*> to_free;
  assert(argc > 0);
  for (int i = 0; i < string_argv.size(); i++) {
    argv[i] = new char[string_argv[i].length()+1];
    strcpy (argv[i], string_argv[i].c_str());
    to_free.push_back(argv[i]);
  }

  // select the right benchmark symbol
  std::function<int(int,char**,int)> func = NULL;
  if (strcmp(argv[0], "parb_sgemm") == 0) {
    std::cout << "main: parboil sgemm" << std::endl;
#ifdef PARBOIL_SGEMM
    func = main_sgemm;
#endif
  } else if (strcmp(argv[0], "parb_stencil") == 0) {
    std::cout << "main: parboil stencil" << std::endl;
#ifdef PARBOIL_STENCIL
    func = main_stencil;
#endif
  } else if (strcmp(argv[0], "cut_sgemm") == 0) {
    std::cout << "main: cutlass sgemm" << std::endl;
#ifdef CUT_SGEMM
    func = main_sgemm;
#endif
  } else if (strcmp(argv[0], "cut_wmma") == 0) {
    std::cout << "main: cutlass wmma" << std::endl;
#ifdef CUT_WMMA
    func = main_wmma;
#endif
  } 
  else {
    std::cout << "Warning: No matching kernels for " << argv[0] << std::endl;
  }

  if (func == NULL) {
    std::cout << "Empty function pointer. Check your compile defines." << std::endl;
    exit(1);
  }

  // invoke the real function
  func(argc, argv, uid);

  // cleanup the char arrays
  for (auto carray: to_free) {
    delete carray;
  }

}


int main(int argc, char** argv) {
  if (argc < 2 || argv[1] == "-h") {
    std::cout << "Usage: ";
    std::cout << "./driver RUNFILE1 [RUNFILE2]" << std::endl;
    abort();
  } 

  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    char* filename = argv[i];

    // extract run arguments from file
    // expect a single line file
    std::string line;
    std::ifstream file (filename);
    if (file.is_open() && std::getline (file,line))
    {
      args.push_back(line);
      file.close();
    } else {
      std::cout << "Error reading file: " << filename << std::endl;
    }
  }

  // spawn threads to invoke separate kernels
  std::thread ts[args.size()];
  for (int i = 0; i < args.size(); i++) {
    shared_push();
    ts[i] = std::thread(invoke, i, args[i]);
  }

  // join threads
  for (auto & t : ts) {
    t.join();
  }

  // sanity check: all flags should now be true
  for (auto f : done_flags) {
    if (!f) std::cout << "Some thread did not set flag to true!!!!" << std::endl;
  }

  cudaDeviceSynchronize();

  return 0;


//  // run the kernels
//  if (strncmp("1",argv[1],1) == 0) {
//    cudaStream_t stream_A;
//    cudaStreamCreate(&stream_A);
//
//    // grab kernel launch and exit function calls from benchmark A and B
//    std::function<int(const int, cudaStream_t &)> kernel_A;
//    std::function<void(void)> cleanup_A;
//
//    invoke(A_str, argc_A, &(argv[A_idx]), kernel_A, cleanup_A);
//
//    kernel_A(1, stream_A);
//
//    cleanup_A();
//
//  } else if (strncmp("2", argv[1], 1) == 0) {
//    cudaStream_t stream_B;
//    cudaStreamCreate(&stream_B);
//
//    // grab kernel launch and exit function calls from benchmark A and B
//    std::function<int(const int, cudaStream_t &)> kernel_B;
//    std::function<void(void)> cleanup_B;
//
//    invoke(B_str, argc_B, &(argv[B_idx]), kernel_B, cleanup_B);
//
//    kernel_B(1, stream_B);
//
//    cleanup_B();
//
//  } else {
//    // run both
//    // create two different cuda streams
//    cudaStream_t stream_A, stream_B;
//
//    cudaStreamCreate(&stream_A);
//    cudaStreamCreate(&stream_B);
//
//    // grab kernel launch and exit function calls from benchmark A and B
//    std::function<int(const int, cudaStream_t &)> kernel_A, kernel_B;
//    std::function<void(void)> cleanup_A, cleanup_B;
//
//    invoke(A_str, argc_A, &(argv[A_idx]), kernel_A, cleanup_A);
//    invoke(B_str, argc_B, &(argv[B_idx]), kernel_B, cleanup_B);
//
//    int iters = 5;
//
//    if (A_str.compare("+spmv") == 0 || B_str.compare("+spmv") == 0) {
//      iters = 30;
//      std::cout << "Launching 30 iters." << std::endl;
//    }
//
//    for (int i = 0; i < iters; i++) {
//      std::cout << "Launching one iteration" << std::endl;
//      kernel_A(1, stream_A);
//      kernel_B(1, stream_B);
//    }
//
//    cleanup_A();
//    cleanup_B();
//

//  }

}
