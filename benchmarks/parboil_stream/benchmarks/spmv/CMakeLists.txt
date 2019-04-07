file(GLOB SPMV_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" src/cuda/*.cu src/cuda/*.cc)
file(GLOB SPMV_COMSRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" common_src/convert-dataset/*.c)


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/common_src/convert-dataset/")

add_library(SPMV ${SPMV_SRC} ${SPMV_COMSRC})

target_link_libraries(SPMV PARBOIL_HELPER)

target_compile_features(SPMV PUBLIC cxx_std_11)
set_target_properties(SPMV PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(SPMV PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(SPMV PROPERTIES LINKER_LANGUAGE CUDA)


