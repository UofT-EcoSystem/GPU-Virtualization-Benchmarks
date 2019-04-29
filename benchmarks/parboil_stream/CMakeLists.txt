cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(parboil LANGUAGES C CXX CUDA)
cmake_policy(SET CMP0076 NEW)


find_package(CUDA)

# by default release build
if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

#set(CUTLASS_NVCC_ARCHS "70" CACHE STRING "The SM architectures to build code for.")
set(CUTLASS_NVCC_EMBED_CUBIN ON CACHE BOOL "Embed compiled CUDA kernel binaries into executables.")
set(CUTLASS_NVCC_EMBED_PTX ON CACHE BOOL "Embed compiled PTX into executables.")

string(APPEND NVCC_FLAGS " --cudart=shared -lcudart")
string(APPEND NVCC_FLAGS " -arch=sm_70")
#string(APPEND NVCC_FLAGS " --default-stream per-thread")

###################################################################################################
#
# Configure CUDA build options
#
###################################################################################################

string(APPEND NVCC_FLAGS_DEBUG " -g")
string(APPEND NVCC_FLAGS_RELWITHDEBINFO " -O3")
string(APPEND NVCC_FLAGS_RELEASE " -O3")

# define NDEBUG for release mode to disable assertions
string(APPEND NVCC_FLAGS_RELEASE " -DNDEBUG")

set(CMAKE_CUDA_FLAGS "${NVCC_FLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE "${NVCC_FLAGS_RELEASE}")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${NVCC_FLAGS_RELWITHDEBINFO}")
set(CMAKE_CUDA_FLAGS_DEBUG "${NVCC_FLAGS_DEBUG}")




add_executable(PAIR pairs/main.cu)


# include cuda dir
target_include_directories(PAIR PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(common)
add_subdirectory(benchmarks)



set_target_properties(PAIR PROPERTIES LINKER_LANGUAGE CUDA)
add_compile_definitions(PARBOIL_STREAM)
target_compile_features(PAIR PUBLIC cxx_std_11)

set_target_properties(PAIR PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(PAIR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
