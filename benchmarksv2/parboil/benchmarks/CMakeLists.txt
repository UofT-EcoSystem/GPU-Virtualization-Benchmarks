target_include_directories(driver PRIVATE ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(sgemm)
add_subdirectory(stencil)

