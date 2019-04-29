target_include_directories(PAIR PRIVATE ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(spmv)
add_subdirectory(sgemm)
add_subdirectory(cutcp)

