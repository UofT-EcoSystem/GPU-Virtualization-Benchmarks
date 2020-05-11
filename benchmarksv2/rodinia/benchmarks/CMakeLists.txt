target_include_directories(driver PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(driver PRIVATE ${CMAKE_CURRENT_LIST_DIR}/common)

target_sources(driver PRIVATE common/rodinia_common.cpp)

if (rod_mummer)
  add_subdirectory(mummergpu)
endif()

if (rod_heartwall)
  add_subdirectory(heartwall)
endif()

if (rod_hotspot)
  add_subdirectory(hotspot)
endif()

if (rod_streamcluster)
  add_subdirectory(streamcluster)
endif()

if (rod_pathfinder)
  add_subdirectory(pathfinder)
endif()

if (rod_lavamd)
  add_subdirectory(lavaMD)
endif()

if (rod_myocyte)
  add_subdirectory(myocyte)
endif()

if (rod_hotspot3d)
  add_subdirectory(hotspot3D)
endif()

if (rod_cfd)
  add_subdirectory(cfd)
endif()
