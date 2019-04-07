#ifndef PARBOIL_INTERFACE_H
#define PARBOIL_INTERFACE_H
  
#include <functional>
#include "cuda_runtime.h"

int main_sgemm (int argc, 
            char** argv,
            std::function<void(const int, cudaStream_t &)> & kernel,
            std::function<void(void)> & exit);

int main_spmv (int argc,
               char** argv,
               std::function<void(const int iter, cudaStream_t & stream)> & kernel,
               std::function<void(void)> & exit);




#endif
