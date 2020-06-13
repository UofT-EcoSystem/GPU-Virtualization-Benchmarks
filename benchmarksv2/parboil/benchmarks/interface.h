#ifndef PARBOIL_INTERFACE_H
#define PARBOIL_INTERFACE_H
  
#ifdef PARBOIL_SGEMM
int main_sgemm (int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef PARBOIL_STENCIL
int main_stencil (int argc, char *argv[], int uid, cudaStream_t & stream);
#endif

#ifdef PARBOIL_LBM
int main_lbm(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef PARBOIL_SPMV
int main_spmv(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef PARBOIL_CUTCP
int main_cutcp(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef PARBOIL_SAD
int main_sad(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef PARBOIL_HISTO
int main_histo(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef PARBOIL_MRIQ
int main_mriq(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef PARBOIL_MRIG
int main_mrig(int argc, char** argv, int uid, cudaStream_t & stream);
#endif


#endif
