/*
 * Synthetic microbenchmarks for integer only instructions
 *
 */

#include <stdio.h>
#include <iostream>
#include <sstream>


#include <cuda_runtime.h>

#define INPUT_LEN 2

__global__ void intadd(unsigned int* in, unsigned int* res)
{
  // in is an array of 6 integers
  int count = 0;
  unsigned int psum1 = 0;
  unsigned int psum2 = 0;
  int id =  blockIdx.x *blockDim.x + threadIdx.x;

#pragma unroll
  while (count < 100000) {
    count++;

    asm("add.u32 %0, %1, %2;": "=r"(psum1): "r"(in[0]), "r"(res[id]));
    asm("add.u32 %0, %1, %2;": "=r"(psum2): "r"(psum1), "r"(in[1]));
    asm("add.u32 %0, %1, %0;": "=r"(res[id]): "r"(psum2), "r"(psum1));

  }
}

int main(int argc, char **argv)
{
  printf("Running synthetic INTEGER only microbenchmarks\n");

  if (argc < INPUT_LEN+1) {
    printf("please input at least %d integers", INPUT_LEN);
    exit(1);
  }

  // invoke kernel
  unsigned int h_in[INPUT_LEN];

  for (int i = 0; i < 2; i++) {
    std::cout << argv[i+1] << std::endl;

    std::stringstream str;
    str << argv[i+1];

    str >> h_in[i];
  }

  const int grid = 432; // 36 SMs * 4 TBs * 3 waves
  const int tb = 256;

  unsigned int h_res[grid*tb];
  memset(h_res, 0, sizeof(unsigned int)*grid*tb);

  unsigned int* d_in;
  unsigned int* d_res;

  cudaMalloc(&d_in, sizeof(unsigned int)*INPUT_LEN);
  cudaMalloc(&d_res, sizeof(unsigned int)*grid*tb);

  cudaMemcpy(d_in, h_in, sizeof(unsigned int)*INPUT_LEN, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_res, sizeof(unsigned int)*grid*tb, cudaMemcpyHostToDevice);

  intadd<<<grid, tb>>>(d_in, d_res);

  cudaMemcpy(h_res, d_res, sizeof(unsigned int)*grid*tb, cudaMemcpyDeviceToHost);

}
