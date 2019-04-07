
file(GLOB SGEMM_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" src/cuda/*.cu src/cuda/*.cc)

add_library(SGEMM ${SGEMM_SRC})

target_link_libraries(SGEMM PARBOIL_HELPER)

target_compile_features(SGEMM PUBLIC cxx_std_11)
set_target_properties(SGEMM PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(SGEMM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(SGEMM PROPERTIES LINKER_LANGUAGE CUDA)

