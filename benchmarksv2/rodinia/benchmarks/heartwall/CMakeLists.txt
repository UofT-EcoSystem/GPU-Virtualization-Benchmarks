target_sources( driver PRIVATE
                main.cu
                AVI/avilib.c
                AVI/avimod.c
              )
target_include_directories(driver PRIVATE AVI)
