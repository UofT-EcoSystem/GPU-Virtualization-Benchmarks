target_include_directories(driver PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(driver PRIVATE 
                          ${CMAKE_CURRENT_LIST_DIR}/cutlass
                          ${CMAKE_CURRENT_LIST_DIR}/gemm-test
                          ${CMAKE_CURRENT_LIST_DIR}/util)
target_sources(driver
  PRIVATE
  cutlass.cu
)
