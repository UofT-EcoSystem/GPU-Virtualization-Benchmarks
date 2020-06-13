target_sources(driver
        PRIVATE
        src/CPU_kernels.c
        src/CUDA_interface.cu
        src/main.cu
        src/scanLargeArray.cu
        src/sort.cu
        )
