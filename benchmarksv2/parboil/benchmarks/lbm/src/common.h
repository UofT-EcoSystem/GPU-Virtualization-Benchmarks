/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef _MAIN_H_
#define _MAIN_H_

#include "cuda_runtime.h"

#ifndef __MCUDA__
#define CUDA_ERRCK                                                      \
{cudaError_t err;                                                     \
  if ((err = cudaGetLastError()) != cudaSuccess) {                    \
    fprintf(stderr, "CUDA error on line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
    exit(-1);                                                         \
  }                                                                   \
}
#else
#define CUDA_ERRCK
#endif


#endif /* _MAIN_H_ */
