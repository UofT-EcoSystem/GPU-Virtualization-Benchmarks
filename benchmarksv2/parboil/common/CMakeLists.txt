

target_sources(driver
		PRIVATE
			src/parboil.c
			src/parboil_cuda.c
)

target_include_directories(driver PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)


