target_include_directories(driver PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(driver PRIVATE ${CMAKE_CURRENT_LIST_DIR}/common/inc)

#target_sources(driver PRIVATE common/src)

if (nvd_dxtc)
  add_subdirectory(dxtc)
endif()

if (nvd_fdtd3d)
  add_subdirectory(FDTD3d)
endif()

if (nvd_blackscholes)
  add_subdirectory(BlackScholes)
endif()

if (nvd_binomial)
  add_subdirectory(binomialOptions)
endif()


