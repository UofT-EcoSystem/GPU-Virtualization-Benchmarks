#
# makefile: Arioc (Linux x64,Power8)
#
# This makefile requires one of the following specified command-line targets:
#
#  AriocE
#  AriocP
#  AriocU
#  clean
#
# An additional phony target (debug or release) may be specified.  If omitted, a debug
#  build is compiled by default.
#

.PHONY: AriocE AriocP AriocU XMC clean debug release

# default build target
all:
	@echo 'Specify one of the following make targets:  AriocE AriocP AriocU clean'
	@echo 'The target name may optionally followed by one of the following:  debug release'
	@echo 'Example:  make AriocP release'

# user-specified build targets
AriocE: ../bin/AriocE

AriocP: ../bin/AriocP

AriocU: ../bin/AriocU

XMC: ../bin/XMC


# look for debug vs release
APPNAME := $(firstword $(MAKECMDGOALS))
BUILDTYPE := $(lastword $(MAKECMDGOALS))

# The debug and release target names are no-ops (in terms of actually building anything).
debug:
        @:

release:
        @:

# CUDA installation path (may override on the command line, e.g. make AriocE CUDA_PATH=/software/apps/cuda/9.2)
CUDA_PATH ?= /usr/local/cuda

# CUDA version-specific
CUDA_INCLUDES := $(CUDA_PATH)/include
CUDA_ARCH := compute_35
CUDA_CODE := \"compute_35,sm_35\"

# tools
#
# g++: We're only interested in compiling for x64, so we use -m64.
# g++: We use -Wno-unknown-pragmas so that g++ doesn't nag about "#pragma region" and other msvc-specific pragmas.
# nvcc: We use -g (generate debug code for host), but we don't use -G (generate debug code for CUDA device) because
#  this breaks Thrust (as of CUDA v6.0 and Thrust 1.7).
#
GPP = g++
NVCC = $(CUDA_PATH)/bin/nvcc

# specific tool configurations for target architectures
# 
TARGET_ARCH := $(shell uname -m)

# x64
ifneq (,$(findstring x86_64,$(TARGET_ARCH)))
ifeq ($(BUILDTYPE),debug)
GPPFLAGS = -std=c++11 -g -c -Wall -Wno-unknown-pragmas -O0 -fno-inline -m64 -D_DEBUG=1 -I $(CUDA_INCLUDES)
NVCCFLAGS = -std=c++11 -g -c -O0 -m64 -gencode arch=$(CUDA_ARCH),code=$(CUDA_CODE) -Xcompiler -O0,-m64 -I $(CUDA_INCLUDES)
else
BUILDTYPE = release
GPPFLAGS = -std=c++11 -g -c -Wall -Wno-unknown-pragmas -O3 -m64 -I $(CUDA_INCLUDES)
NVCCFLAGS = -std=c++11 -g -c -O3 -m64 -gencode arch=$(CUDA_ARCH),code=$(CUDA_CODE) -Xcompiler -O3,-m64 -I $(CUDA_INCLUDES)
endif
endif

# ppc64
ifneq (,$(findstring ppc64,$(TARGET_ARCH)))
ifeq ($(BUILDTYPE),debug)
GPPFLAGS = -std=c++11 -g -c -Wall -Wno-unknown-pragmas -O0 -fno-inline -D_DEBUG=1 -I $(CUDA_INCLUDES)
NVCCFLAGS = -std=c++11 -g -c -O0 -m64 -gencode arch=$(CUDA_ARCH),code=$(CUDA_CODE) -Xcompiler -O0,-m64 -I $(CUDA_INCLUDES)
else
BUILDTYPE = release
GPPFLAGS = -std=c++11 -g -c -Wall -Wno-unknown-pragmas -O3 -I $(CUDA_INCLUDES)
NVCCFLAGS = -std=c++11 -g -c -O3 -m64 -gencode arch=$(CUDA_ARCH),code=$(CUDA_CODE) -Xcompiler -O3,-m64,-mcpu=power8,-mtune=power8, -I $(CUDA_INCLUDES)
endif
endif


# Initializations specific to each target:
#
# A list of the subdirectories that contain the source code used for the build target.
# Targets that use the CUDA libraries are linked with nvcc instead of g++.
#
ifeq ($(APPNAME),AriocE)
SUBDIRS = Windux CppCommon TaskUnit CudaCommon AriocCommon AriocE
LD = $(NVCC)
LDFLAGS = --cudart static -link
endif

ifeq ($(APPNAME),AriocP)
SUBDIRS = Windux CppCommon TaskUnit CudaCommon AriocCommon AriocBase AriocP
LD = $(NVCC)
LDFLAGS = --cudart static -link
endif

ifeq ($(APPNAME),AriocU)
SUBDIRS = Windux CppCommon TaskUnit CudaCommon AriocCommon AriocBase AriocU
LD = $(NVCC)
LDFLAGS = --cudart static -link
endif

ifeq ($(APPNAME),XMC)
SUBDIRS = Windux CppCommon TaskUnit AriocCommon XMC
LD = g++
LDFLAGS =
endif


# non-default link libraries
LIBS = -lpthread


# Each subdirectory contains a file named subdir.mk that looks like this:
#    CPPS += $(wildcard subdirname/*.cpp)
# where subdirname is the name of the subdirectory
#
# If there are .cu files in the subdirectory, there's also
#    CUS += $(wildcard subdirname/*.cu)
# So the result is that CPPS and CUS are set up to contain a list of .cpp and .cu filenames respectively
#  (prefixed by their subdirectory name) for all of the subdirectories in SUBDIRS
#
CPPS :=
CUS :=
include $(patsubst %,%/subdir.mk,$(SUBDIRS))

# create a list of .o files that correspond to the list of .cpp files
CPPOBJS := $(patsubst %.cpp,%.o, $(filter %.cpp,$(CPPS)))

# create a list of .gpu.o files that correspond to the list of .cu files
CUOBJS := $(patsubst %.cu,%.gpu.o, $(filter %.cu,$(CUS)))

# link the objects (which differ for each target executable)
../bin/$(APPNAME): $(CPPOBJS) $(CUOBJS)
	mkdir -p ../bin
	$(LD) $(LDFLAGS) $(LIBS) -o $@ $(CPPOBJS) $(CUOBJS)
	@echo 'Build complete for $@ ($(BUILDTYPE))'


# Include (in this makefile) dependency info for existing .o and .gpu.o files.  The dash preceding the
# include keyword tells make not to emit a warning if the .d or .gpu.d file does not exist.
-include $(CPPOBJS:.o=.d)
-include $(CUOBJS:.gpu.o=.gpu.d)

# Compile .cpp files and generate dependency info.
#
# This rule takes precedence over make's builtin default invocation of gcc for generating .o files from
# .cpp files.
#
# The horrid-looking garble in the sed expression works around the way gcc emits rules.  A typical example
#  from gcc using the -MM flag is:
#       Windux.o: Windux/Windux.cpp Windux/Windux.h
# The regex (delimited by tilde) replaces everything up to the colon with the current make target, so we
#  end up with:
#       Windux/Windux.o: Windux/Windux.cpp Windux/Windux.h
#
%.d %.o : %.cpp
	$(GPP) $(GPPFLAGS) -o $*.o $<
	$(GPP) -MM $*.cpp | sed -e 's~^\(.*\)\.o:~$@:~' > $*.d

# Same thing for .cu files except...
#  We need a .gpu.o and a .gpu.d file for the target, so within the sed expression we use the "stem" shorthand
#   ($*, i.e., everything that corresponds to the wildcard part of the target specification) and append ".gpu.d".
#
# As of version 6.0, nvcc still supports only -M, which gives a dependency list that contains system and
#  CUDA include files.  We're not interested in those, so we use g++ instead (but g++ doesn't know that
#  .cu files contain C++, hence the -x flag.)
#
%.gpu.d %.gpu.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $*.gpu.o $<
	$(GPP) -MM -xc++ $*.cu | sed -e 's~^\(.*\)\.o:~$*.gpu.o:~' > $*.gpu.d



# The following rule deletes the following files in all subdirectories:
#  .o
#  .d
#
# It also cleans up the following files used only in Visual Studio for Windows builds:
#  *.vcxproj
#  *.vcxproj.filters
#  *.vcxproj.user
#  Arioc.sdf
#  Arioc.sln
#  Arioc.suo
#  targetver.h
#
# It works as follows:
#  find . -name finds all the files with the specified pattern; the -print0 switch causes the output to be generated
#   with a null separator for the benefit of xarg -0.
#  xarg parses its input and uses each parsed token as an argument to what follows (i.e, rm get invoked once for each
#   parsed token).  The -0 flag indicates that the input is separated with nulls.
#
# (Invoke with: make clean)
clean:
	find . -name '*.o' -print0 | xargs -0 rm -v -f
	find . -name '*.d' -print0 | xargs -0 rm -v -f
	find . -name '*.vcxproj' -print0 | xargs -0 rm -v -f
	find . -name '*.vcxproj.filters' -print0 | xargs -0 rm -v -f
	find . -name '*.vcxproj.user' -print0 | xargs -0 rm -v -f
	find . -name 'targetver.h' -print0 | xargs -0 rm -v -f
	rm -v -f Arioc.sdf
	rm -v -f Arioc.sln
	rm -v -f Arioc.suo

