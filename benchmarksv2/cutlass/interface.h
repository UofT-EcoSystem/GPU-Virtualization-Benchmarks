#ifndef CUTLASS_INTERFACE_H
#define CUTLASS_INTERFACE_H

int main_sgemm (int argc, char** argv, int uid, cudaStream_t & stream);
int main_wmma (int argc, char *argv[], int uid, cudaStream_t & stream);

#endif
