target_include_directories(driver PRIVATE ${CMAKE_CURRENT_LIST_DIR})

if (parb_sgemm)
  add_subdirectory(sgemm)
endif()

if (parb_stencil)
  add_subdirectory(stencil)
endif()

if (parb_lbm)
  add_subdirectory(lbm)
endif()

if (parb_spmv)
  add_subdirectory(spmv)
endif()

if (parb_cutcp)
  add_subdirectory(cutcp)
endif()

if (parb_sad)
  add_subdirectory(sad)
endif()

if (parb_histo)
  add_subdirectory(histo)
endif()

if (parb_mriq)
  add_subdirectory(mri-q)
endif()

if (parb_mrig)
  add_subdirectory(mri-gridding)
endif()

if (parb_tpacf)
  add_subdirectory(tpacf)
endif()

if (parb_bfs)
  add_subdirectory(bfs)
endif()
