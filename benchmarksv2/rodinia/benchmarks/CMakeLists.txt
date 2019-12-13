target_include_directories(driver PRIVATE ${CMAKE_CURRENT_LIST_DIR})

if (rod_mummer)
  add_subdirectory(mummergpu)
endif()

if (rod_heartwall)
  add_subdirectory(heartwall)
endif()

if (rod_hotspot)
  add_subdirectory(hotspot)
endif()


