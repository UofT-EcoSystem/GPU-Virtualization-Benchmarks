# (c) 2007 The Board of Trustees of the University of Illinois.

# Cuda-related definitions common to all benchmarks

########################################
# Variables
########################################

# c.default is the base along with CUDA configuration in this setting
include $(PARBOIL_ROOT)/common/platform/c.default.mk

# Paths
include ../../../../scripts/config
# CUDAHOME=/usr/local/cuda-9.0

# Programs
CUDACC=$(CUDAHOME)/bin/nvcc
CUDALINK=$(CUDAHOME)/bin/nvcc

# Flags
PLATFORM_CUDACFLAGS=-O3
PLATFORM_CUDALDFLAGS=-lm -lpthread

APP_CUDACFLAGS += $(arch)

