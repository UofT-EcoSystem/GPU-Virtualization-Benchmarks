target_sources(driver PRIVATE
              src/mummergpu_main.cpp
              src/mummergpu.cu
              src/mummergpu_gold.cpp
              src/PoolMalloc.cpp
              src/smith-waterman.cpp
              src/suffix-tree.cpp
              )

target_include_directories(driver PRIVATE src)

