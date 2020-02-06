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

// user includes
#include "parboil/benchmarks/interface.h"
#include "cutlass/interface.h"
#include "rodinia/benchmarks/interface.h"
#include "nvidia/interface.h"

std::vector<bool> done_flags;
std::vector<bool> start_flags;
std::mutex lock_flag;
std::mutex lock_flag2;

bool set_and_check(int uid, bool start) {
  // this function is guarded by the mutex
  std::lock_guard<std::mutex> guard(lock_flag);

  if (!start) {
    if (uid < done_flags.size()) {
      done_flags[uid] = true;
    }


    for (auto f : done_flags) {
      if (!f) return false;
    }

    return true;
  } 
  else {
    if (uid < start_flags.size()) {
      start_flags[uid] = true;
    }


    for (auto f : start_flags) {
      if (!f) return false;
    }

    return true;
  }
}

void shared_push() {
  // this function is guarded by the mutex
  std::lock_guard<std::mutex> guard(lock_flag);
  done_flags.push_back(false);
  start_flags.push_back(false);
}


void invoke(int uid, std::string kernel_arg, cudaStream_t* stream)
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
  std::function<int(int,char**,int,cudaStream_t&)> func = NULL;
  if (strcmp(argv[0], "parb_sgemm") == 0) {
    std::cout << "main: parboil sgemm" << std::endl;
#ifdef PARBOIL_SGEMM
    func = main_sgemm;
#endif
  } 
  else if (strcmp(argv[0], "parb_stencil") == 0) {
    std::cout << "main: parboil stencil" << std::endl;
#ifdef PARBOIL_STENCIL
    func = main_stencil;
#endif
  } 
  else if (strcmp(argv[0], "parb_lbm") == 0) {
    std::cout << "main: parboil lbm" << std::endl;
#ifdef PARBOIL_LBM
    func = main_lbm;
#endif
  } 
  else if (strcmp(argv[0], "parb_spmv") == 0) {
    std::cout << "main: parboil spmv" << std::endl;
#ifdef PARBOIL_SPMV
    func = main_spmv;
#endif
  } 
  else if (strcmp(argv[0], "parb_cutcp") == 0) {
    std::cout << "main: parboil cutcp" << std::endl;
#ifdef PARBOIL_CUTCP
    func = main_cutcp;
#endif
  } 
  else if (strcmp(argv[0], "cut_sgemm") == 0) {
    std::cout << "main: cutlass sgemm" << std::endl;
#ifdef CUT_SGEMM
    func = main_sgemm;
#endif
  } 
  else if (strcmp(argv[0], "cut_wmma") == 0) {
    std::cout << "main: cutlass wmma" << std::endl;
#ifdef CUT_WMMA
    func = main_wmma;
#endif
  } 
  else if (strcmp(argv[0], "rod_mummer") == 0) {
    std::cout << "main: rodinia mummer" << std::endl;
#ifdef RODINIA_MUMMER
    func = main_mummer;
#endif
  }
  else if (strcmp(argv[0], "rod_heartwall") == 0) {
    std::cout << "main: rodinia heartwall" << std::endl;
#ifdef RODINIA_HEARTWALL
    func = main_heartwall;
#endif
  }
  else if (strcmp(argv[0], "rod_hotspot") == 0) {
    std::cout << "main: rodinia hotspot" << std::endl;
#ifdef RODINIA_HOTSPOT
    func = main_hotspot;
#endif
  }
  else if (strcmp(argv[0], "rod_streamcluster") == 0) {
    std::cout << "main: rodinia streamcluster" << std::endl;
#ifdef RODINIA_STREAMCLUSTER
    func = main_streamcluster;
#endif
  }
  else if (strcmp(argv[0], "rod_pathfinder") == 0) {
    std::cout << "main: rodinia pathfinder" << std::endl;
#ifdef RODINIA_PATHFINDER
    func = main_pathfinder;
#endif
  }
  else if (strcmp(argv[0], "rod_lavamd") == 0) {
    std::cout << "main: rodinia lavamd" << std::endl;
#ifdef RODINIA_LAVAMD
    func = main_lavamd;
#endif
  }
  else if (strcmp(argv[0], "rod_myocyte") == 0) {
    std::cout << "main: rodinia myocyte" << std::endl;
#ifdef RODINIA_MYOCYTE
    func = main_myocyte;
#endif
  }
  else if (strcmp(argv[0], "rod_hotspot3d") == 0) {
    std::cout << "main: rodinia hotspot3d" << std::endl;
#ifdef RODINIA_HOTSPOT3D
    func = main_hotspot3d;
#endif
  }
  else if (strcmp(argv[0], "nvd_dxtc") == 0) {
    std::cout << "main: nvidia dxtc" << std::endl;
#ifdef NVD_DXTC
    func = main_dxtc;
#endif
  }
  else if (strcmp(argv[0], "nvd_fdtd3d") == 0) {
    std::cout << "main: nvidia fdtd3d" << std::endl;
#ifdef NVD_FDTD3D
    func = main_fdtd3d;
#endif
  }
  else if (strcmp(argv[0], "nvd_blackscholes") == 0) {
    std::cout << "main: nvidia blackscholes" << std::endl;
#ifdef NVD_BLACKSCHOLES
    func = main_blackscholes;
#endif
  }
  else if (strcmp(argv[0], "nvd_binomial") == 0) {
    std::cout << "main: nvidia binomial" << std::endl;
#ifdef NVD_BINOMIAL
    func = main_binomial;
#endif
  }
  else if (strcmp(argv[0], "nvd_sobol") == 0) {
    std::cout << "main: nvidia sobol" << std::endl;
#ifdef NVD_SOBOL
    func = main_sobol;
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
  func(argc, argv, uid, *stream);

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
  cudaStream_t streams[args.size()];
  for (int i = 0; i < args.size(); i++) {
    shared_push();
    cudaStreamCreate(&(streams[i]));

    ts[i] = std::thread(invoke, i, args[i], &(streams[i]));
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
}
