//
// Created by Serina Tan on 2020-06-09.
//

#ifndef PARBOIL_FRAMEWORK_H
#define PARBOIL_FRAMEWORK_H

#include <vector>
#include <string>

#include "cuda_runtime_api.h"

struct app_t {
  app_t() {
    pFunc = NULL;
    num_repeat = 1;

    current_pos = 0;
    start = false;
    done = false;
  }

  std::function<int(int,char**,int,cudaStream_t&)> pFunc;
  std::vector<std::string> params;
  unsigned num_repeat;

  // Current_pos is incremented when set_and_check with start==false is called.
  unsigned current_pos;

  // Only used for non-synthetic workloads
  bool start;
  bool done;
};

struct stream_ops_t {
  stream_ops_t() {
    cudaStreamCreate(&cudaStream);
    current_pos = 0;
    done_iteration = false;
    exited = false;
  }

  std::vector<app_t> apps;
  cudaStream_t cudaStream;

  unsigned current_pos;
  bool done_iteration;
  bool exited;
};

#endif // PARBOIL_FRAMEWORK_H
