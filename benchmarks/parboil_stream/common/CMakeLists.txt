

target_sources(PAIR
		PRIVATE
			src/parboil.c
			src/parboil_cuda.c
)

target_include_directories(PAIR PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)


