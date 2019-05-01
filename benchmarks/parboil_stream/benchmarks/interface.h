#ifndef PARBOIL_INTERFACE_H
#define PARBOIL_INTERFACE_H
  
#include <functional>
#include "cuda_runtime.h"

int main_sgemm (int argc, 
            char** argv,
            std::function<int(const int, cudaStream_t &)> & kernel,
            std::function<void(void)> & cleanup);

int main_spmv (int argc,
               char** argv,
               std::function<int(const int iter, cudaStream_t & stream)> & kernel,
               std::function<void(void)> & cleanup);

int main_cutcp (int argc,
               char** argv,
               std::function<int(const int iter, cudaStream_t & stream)> & kernel,
               std::function<void(void)> & cleanup);

int main_mriq (int argc, 
           char *argv[],
           std::function<int(const int, cudaStream_t &)> & kernel,
           std::function<void(void)> & cleanup);



int main_tpacf (int argc, 
           char *argv[],
           std::function<int(const int, cudaStream_t &)> & kernel,
           std::function<void(void)> & cleanup);

int main_lbm (int argc, 
           char *argv[],
           std::function<int(const int, cudaStream_t &)> & kernel,
           std::function<void(void)> & cleanup);


int main_sad (int argc, 
           char *argv[],
           std::function<int(const int, cudaStream_t &)> & kernel,
           std::function<void(void)> & cleanup);


int main_stencil (int argc, 
           char *argv[],
           std::function<int(const int, cudaStream_t &)> & kernel,
           std::function<void(void)> & cleanup);
#endif
