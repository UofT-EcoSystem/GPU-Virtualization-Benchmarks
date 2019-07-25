#!/bin/bash

# run.sh dgemm|wmma|sgemm|igemm nsight|sim

NSIGHT=/usr/local/cuda/NsightCompute-2019.3/nv-nsight-cu-cli
SECTION_FOLDER=/home/serinatan/project/GPU-Virtualization-Benchmarks/benchmarks/sections
NVPROF=/usr/local/cuda/bin/nvprof

if [ $2 == "nsight" ]; then
  PREFIX="$NSIGHT --section-folder $SECTION_FOLDER --section InstCounter --csv -f"
elif [ $2 == "nvprof" ]; then
  PREFIX="$NVPROF --print-gpu-trace --csv -f "
else
  PREFIX=""
fi

if [ $1 == "dgemm" ]; then
  make clean
  make CFLAGS=-DDGEMM_4
  $PREFIX ./cutlass-test | tee profile/dgemm4$2.txt
elif [ $1 == "wmma" ]; then
  make clean
  make CFLAGS=-DWMMA_36
  $PREFIX ./cutlass-test | tee profile/wmma36$2.txt
elif [ $1 == "sgemm" ]; then
  make clean
  make CFLAGS=-DSGEMM_4
  $PREFIX ./cutlass-test | tee profile/sgemm4$2.txt
elif [ $1 == "igemm" ]; then
  make clean
  make CFLAGS=-DIGEMM_1
  $PREFIX ./cutlass-test | tee profile/igemm1$2.txt
elif [ $1 == "hgemm" ]; then
  make clean
  make CFLAGS=-DHGEMM_1
  $PREFIX ./cutlass-test | tee profile/hgemm1$2.txt
fi

