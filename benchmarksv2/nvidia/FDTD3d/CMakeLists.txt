target_sources(driver PRIVATE
               src/FDTD3d.cpp
               src/FDTD3dGPU.cu
               src/FDTD3dReference.cpp
              )
target_include_directories(driver PRIVATE inc)
