target_sources(driver PRIVATE main.cu
                              kernel/kernel_gpu_cuda_wrapper.cu
                              util/timer/timer.c
                              util/num/num.c
                              util/device/device.cu)
