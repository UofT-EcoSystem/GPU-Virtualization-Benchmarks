#ifndef PARBOIL_INTERFACE_H
#define PARBOIL_INTERFACE_H
  
int main_sgemm (int argc, char** argv, int uid, cudaStream_t & stream);
int main_stencil (int argc, char *argv[], int uid, cudaStream_t & stream);
int main_lbm(int argc, char** argv, int uid, cudaStream_t & stream);

#endif
