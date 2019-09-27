target_sources(driver PRIVATE 
              src/main.cu
              src/file.cc
              src/gpu_info.cc
              common_src/convert_dataset.c
              common_src/mmio.c)
target_include_directories(driver PRIVATE common_src)
