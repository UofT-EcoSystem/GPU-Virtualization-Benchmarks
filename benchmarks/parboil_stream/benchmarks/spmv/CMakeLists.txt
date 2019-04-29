
target_sources(PAIR
		PRIVATE
			common_src/convert-dataset/convert_dataset.c
			common_src/convert-dataset/mmio.c
)

target_sources(PAIR
		PRIVATE
			src/cuda/file.cc
			src/cuda/gpu_info.cc
			src/cuda/jds_kernels.cu
			src/cuda/main.cu
)

target_include_directories(PAIR PRIVATE ${CMAKE_CURRENT_LIST_DIR}/common_src/convert-dataset)
