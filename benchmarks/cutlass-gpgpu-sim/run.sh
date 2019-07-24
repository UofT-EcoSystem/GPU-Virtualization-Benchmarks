#!/bin/bash

# run.sh dgemm|wmma|sgemm|igemm nsight|sim

NSIGHT=/usr/local/cuda-10.1/NsightCompute-2019.3/nv-nsight-cu-cli
SECTION_FOLDER=/home/serinatan/project/GPU-Virtualization-Benchmarks/benchmarks/sections

if [ $2 == "nsight" ]; then
  PREFIX="$NSIGHT --section-folder $SECTION_FOLDER --section InstCounter --csv"
else
  PREFIX=""
fi

if [ $1 == "dgemm" ]; then
  make clean
  make CFLAGS=-DDGEMM_3
  $PREFIX ./cutlass-test | tee profile/dgemm3.txt
elif [ $1 == "wmma" ]; then
  make clean
  make CFLAGS=-DWMMA_36
  $PREFIX ./cutlass-test | tee profile/wmma36.txt
elif [ $1 == "sgemm" ]; then
  make clean
  make CFLAGS=-DSGEMM_3
  $PREFIX ./cutlass-test | tee profile/sgemm3.txt
fi

