#ifndef PARBOIL_INTERFACE_H
#define PARBOIL_INTERFACE_H
  
#include <functional>
#include "cuda_runtime.h"

int main_sgemm (int argc, char** argv, int uid);
int main_stencil (int argc, char *argv[], int uid);

#endif
