cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(parboil LANGUAGES C CXX CUDA)

find_package(CUDA)

# by default release build
if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

set(CUTLASS_NVCC_ARCHS "75" CACHE STRING "The SM architectures to build code for.")
set(CUTLASS_NVCC_EMBED_CUBIN ON CACHE BOOL "Embed compiled CUDA kernel binaries into executables.")
set(CUTLASS_NVCC_EMBED_PTX ON CACHE BOOL "Embed compiled PTX into executables.")

string(APPEND NVCC_FLAGS " --cudart=shared -lcudart")
string(APPEND NVCC_FLAGS " --default-stream per-thread")

###################################################################################################
#
# Configure CUDA build options
#
###################################################################################################

# Set NVCC arguments
foreach(ARCH ${CUTLASS_NVCC_ARCHS})
  if(CUTLASS_NVCC_EMBED_CUBIN)
    string(APPEND NVCC_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
  endif()
  if(CUTLASS_NVCC_EMBED_PTX)
    string(APPEND NVCC_FLAGS " -gencode arch=compute_${ARCH},code=compute_${ARCH}")
  endif()
endforeach()

string(APPEND NVCC_FLAGS_DEBUG " -g")
string(APPEND NVCC_FLAGS_RELWITHDEBINFO " -O3")
string(APPEND NVCC_FLAGS_RELEASE " -O3")

# define NDEBUG for release mode to disable assertions
string(APPEND NVCC_FLAGS_RELEASE " -DNDEBUG")

set(CMAKE_CUDA_FLAGS "${NVCC_FLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE "${NVCC_FLAGS_RELEASE}")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${NVCC_FLAGS_RELWITHDEBINFO}")
set(CMAKE_CUDA_FLAGS_DEBUG "${NVCC_FLAGS_DEBUG}")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/common/include") 
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/spmv/common_src/convert-dataset")

# include cuda dir
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})




# hacky shit: just glob all the files and compile them together
#file(GLOB HELPER RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} common/src/*.c)
#file(GLOB SGEMM RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} benchmarks/sgemm/src/cuda/*.cu benchmarks/sgemm/src/cuda/*.cc)
#file(GLOB SPMV RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} benchmarks/spmv/src/cuda/*.cu benchmarks/spmv/src/cuda/*.cc)
#file(GLOB SPMV_COM RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} benchmarks/spmv/common_src/convert-dataset/*.c)



#add_executable(PAIR ${HELPER} ${SGEMM} ${SPMV} ${SPMV_COM} pairs/main.cu)

#target_compile_features(PAIR PUBLIC cxx_std_11)

add_subdirectory(common)
add_subdirectory(benchmarks)
add_subdirectory(pairs)
