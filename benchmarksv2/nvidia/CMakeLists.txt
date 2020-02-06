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

if (nvd_sobol)
  add_subdirectory(SobolQRNG)
endif()

if (nvd_stereo)
  add_subdirectory(stereoDisparity)
endif()

if (nvd_interval)
  add_subdirectory(interval)
endif()


