cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(parboil LANGUAGES C CXX CUDA)
cmake_policy(SET CMP0076 NEW)

option(parb_sgemm "compiling sgemm kennels" OFF)
option(parb_stencil "compiling stencil kennels" OFF)
option(parb_lbm "compiling lbm kennels" OFF)
option(parb_cutcp "compiling cutcp kennels" OFF)
option(cut_sgemm "compiling cutlass sgemm kernels" OFF)
option(cut_wmma "compiling cutlass wmma kernels" OFF)
option(rod_mummer "compiling rodinia mummer kernels" OFF)
option(rod_heartwall "compiling rodinia heartwall kernels" OFF)
option(rod_hotspot "compiling rodinia hotspot kernels" OFF)
option(rod_streamcluster "compiling rodinia streamcluster kernels" OFF)


find_package(OpenMP)

# by default release build
if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

##########################################################################################
#
# Configure CUDA build options
#
##########################################################################################
#set(CUTLASS_NVCC_ARCHS "70" CACHE STRING "The SM architectures to build code for.")
set(CUTLASS_NVCC_EMBED_CUBIN ON CACHE BOOL "Embed compiled CUDA kernel binaries into executables.")
set(CUTLASS_NVCC_EMBED_PTX ON CACHE BOOL "Embed compiled PTX into executables.")

string(APPEND NVCC_FLAGS " --cudart=shared -lcudart")
string(APPEND NVCC_FLAGS " -arch=sm_70")
# gpgpusim does not support this stream mode
#string(APPEND NVCC_FLAGS " --default-stream per-thread")

string(APPEND NVCC_FLAGS_DEBUG " -g")
string(APPEND NVCC_FLAGS_RELWITHDEBINFO " -O3")
string(APPEND NVCC_FLAGS_RELEASE " -O3")

# define NDEBUG for release mode to disable assertions
string(APPEND NVCC_FLAGS_RELEASE " -DNDEBUG")

set(CMAKE_CUDA_FLAGS "${NVCC_FLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE "${NVCC_FLAGS_RELEASE}")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${NVCC_FLAGS_RELWITHDEBINFO}")
set(CMAKE_CUDA_FLAGS_DEBUG "${NVCC_FLAGS_DEBUG}")

# add benchmark source
add_executable(driver main.cu)

add_subdirectory(parboil)
add_subdirectory(cutlass)
add_subdirectory(rodinia)

# include cuda dir
target_include_directories(driver PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(parb_sgemm)
  MESSAGE ("cmake: sgemm.")
  target_compile_definitions(driver PRIVATE PARBOIL_SGEMM)
endif()

if(parb_stencil)
  MESSAGE ("cmake: stencil.")
  target_compile_definitions(driver PRIVATE PARBOIL_STENCIL)
endif()

if(parb_lbm)
  MESSAGE ("cmake: lbm.")
  target_compile_definitions(driver PRIVATE PARBOIL_LBM)
endif()

if(parb_spmv)
  MESSAGE ("cmake: spmv.")
  target_compile_definitions(driver PRIVATE PARBOIL_SPMV)
endif()

if(parb_cutcp)
  MESSAGE ("cmake: cutcp.")
  target_compile_definitions(driver PRIVATE PARBOIL_CUTCP)
endif()

if(cut_sgemm)
  MESSAGE ("cmake: cutlass sgemm.")
  target_compile_definitions(driver PRIVATE CUT_SGEMM)
endif()

if(cut_wmma)
  MESSAGE ("cmake: cutlass wmma.")
  target_compile_definitions(driver PRIVATE CUT_WMMA)
endif()

if(rod_mummer)
  MESSAGE ("cmake: rodinia mummer.")
  target_compile_definitions(driver PRIVATE RODINIA_MUMMER)
endif()

if(rod_heartwall)
  MESSAGE ("cmake: rodinia heartwall.")
  target_compile_definitions(driver PRIVATE RODINIA_HEARTWALL)
endif()

if(rod_hotspot)
  MESSAGE ("cmake: rodinia hotspot.")
  target_compile_definitions(driver PRIVATE RODINIA_HOTSPOT)
endif()

if(rod_streamcluster)
  MESSAGE ("cmake: rodinia streamcluster.")
  target_compile_definitions(driver PRIVATE RODINIA_STREAMCLUSTER)
endif()

set_target_properties(driver PROPERTIES LINKER_LANGUAGE CUDA)
add_compile_definitions(PARBOIL_STREAM)
target_compile_features(driver PUBLIC cxx_std_11)

set(CMAKE_CUDA_LINK_EXECUTABLE "<CMAKE_CUDA_COMPILER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

#target_compile_options(driver PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(driver PRIVATE pthread)
target_link_options(driver PRIVATE -Xlinker -lgomp)


#set_target_properties(PAIR PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(PAIR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
