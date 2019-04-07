add_executable(PAIR main.cu)
target_link_libraries(PAIR PRIVATE SGEMM SPMV)

target_compile_features(PAIR PUBLIC cxx_std_11)
#set_target_properties(PAIR PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(PAIR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(PAIR PROPERTIES LINKER_LANGUAGE CXX)
