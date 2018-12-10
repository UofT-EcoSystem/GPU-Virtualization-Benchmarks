# Benchmarks for GPU Virtualization 
This repo is a suite of GPGPU benchmarks that will be used to collect both motivation data for GPU resource virtualization and to evaluate our proposed solutions. 

## List of NVPROF Metrics Considered

* DP util: double precision function unit utilization on a scale of 0 to 10 [double_precision_fu_utilization]
* DP effic: ratio of achieved to peak double-precision FP operations [flop_dp_efficiency]
* SP util: [single_precision_fu_utilization]
* SP effic: [flop_sp_efficiency]
* HP util: [half_precision_fu_utilization]
* HP effic: [flop_hp_efficiency]
* DRAM util: [dram_utilization]
* DRAM read throughput: [dram_read_throughput]
* DRAM write throughput: [dram_write_throughput]
* L1/tex hit rate: Hit rate for global load and store in unified l1/tex cache [global_hit_rate]
* L2 hit rate: Hit rate at L2 cache for all requests from texture cache [l2_tex_hit_rate] (not avail. on Volta)
* Shared memory util: on a scale of 0 to 10 [shared_utilization]
* Special func unit util: on a scale of 0 to 10 [special_fu_utilization]
* tensor precision util: [tensor_precision_fu_utilization] (not avail. on CUDA 9.0)
* tensor int8 util: [tensor_int_fu_utilization] (not avail. on CUDA 9.0)


## List of Benchmarks

<p align="center">
 Table 1 Individual Benchmark Characteristics
</p>

|  Source  | Application | Benchmark Name | C/M Bound | DP Util| DP effic | SP Util | SP effic | HP Util | HP effic | DRAM Util | DRAM read thruput | DRAM write thruput | L1/tex hit rate  | L2 hit rate | Shared memory util | Special func unit util | tensor FP util | tensor int8 util | sm effic | 
| ----- | ---- | -- | -- | --- | -- | -- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---| ---| 
|Cutlass | SP matrix multiply | cutlass_sgemm_256 | C | Idle (0) | 0.00% | High (7) | 3.56% | Idle (0) | 0.00% | Low (1) | 7.62GB/s | 15.15GB/s | 0.00% | N/A | Low (1) | Idle (0) | N/A | N/A | 4.75% | 
|Cutlass | SP matrix multiply | cutlass_sgemm_512 | C | Idle (0) | 0.00% | High (8) | 15.74% | Idle (0) | 0.00% | Low (1) | 16.06GB/s | 20.95GB/s | 0.00% | N/A | Low (1) | Idle (0) | N/A | N/A | 19.45% | 
|Cutlass | SP matrix multiply | cutlass_sgemm_1024 | C | Idle (0) | 0.00% | High (8) | 64.51% | Idle (0) | 0.00% | Low (1) | 31.66GB/s | 25.81GB/s | 0.00% | N/A | Low (2) | Idle (0) | N/A | N/A | 78.38% | 
|Cutlass | SP matrix multiply | cutlass_sgemm_2048 | C | Idle (0) | 0.00% | Max (10) | 73.23% | Idle (0) | 0.00% | Low (1) | 37.95GB/s | 9.16GB/s | 1.30% | N/A | Low (3) | Idle (0) | N/A | N/A | 81.37% | 
|Cutlass | SP matrix multiply | cutlass_sgemm_4096 | C | Idle (0) | 0.00% | Max (10) | 90.34% | Idle (0) | 0.00% | Low (2) | 61.49GB/s | 5.61GB/s | 0.41% | N/A | Low (3) | Idle (0) | N/A | N/A | 98.23% | 
|CUDA SDK | HP matrix multiply | tensor_gemm | C | Idle (0) | 0.00% | Low (1) | 0.08% | Idle (0) | 0.00% | Mid (4) | 184.41GB/s | 21.12GB/s | 0.00% | N/A | Mid (5) | Low (1) | N/A | N/A | 96.74% | 
|Cutlass | INT matrix multiply | cutlass_igemm_256 | C | Idle (0) | 0.00% | Mid (6) | 0.00% | Idle (0) | 0.00% | Low (1) | 5.40GB/s | 20.94GB/s | 0.00% | N/A | Low (1) | Idle (0) | N/A | N/A | 4.30% | 
|Cutlass | INT matrix multiply | cutlass_igemm_512 | C | Idle (0) | 0.00% | Mid (6) | 0.00% | Idle (0) | 0.00% | Low (1) | 13.60GB/s | 26.52GB/s | 0.00% | N/A | Low (1) | Idle (0) | N/A | N/A | 18.42% | 
|Cutlass | INT matrix multiply | cutlass_igemm_1024 | C | Idle (0) | 0.00% | High (7) | 0.00% | Idle (0) | 0.00% | Low (2) | 27.60GB/s | 57.73GB/s | 0.00% | N/A | Low (2) | Idle (0) | N/A | N/A | 75.14% | 
|Cutlass | INT matrix multiply | cutlass_igemm_2048 | C | Idle (0) | 0.00% | High (9) | 0.00% | Idle (0) | 0.00% | Low (1) | 31.76GB/s | 44.74GB/s | 1.32% | N/A | Low (2) | Idle (0) | N/A | N/A | 81.01% | 
|Cutlass | INT matrix multiply | cutlass_igemm_4096 | C | Idle (0) | 0.00% | High (9) | 0.00% | Idle (0) | 0.00% | Low (2) | 81.71GB/s | 26.73GB/s | 0.05% | N/A | Low (3) | Idle (0) | N/A | N/A | 98.16% | 
| Parboil | Sparse vector matrix multiply | parboil_spmv | M | Idle (0) | 0.00% | Low (1) | 0.85% | Idle (0) | 0.00% | High (9) | 464.77GB/s | 37.25GB/s | 48.81% | N/A | Idle (0) | Idle (0) | N/A | N/A | 90.80% |



## To Build and Run Benchmarks

TODO...

## To Add New Benchmarks
To use our benchmark wrapper script, each benchmark should implement the same interface to synchronize kernel launches. Consider a test comprises two applications A and B, with each application invoking different GPU compute kernels. The goal is to capture performance data during the period where both kernels from A and B execute concurrently. Hence, kernel execution in each application should start at the same time and ideally end around the same time. Since it's impossible to guarantee all kernels end simultaneously due to kernel runtime difference, the data processing script will calculate the delta of kernel elapased time between A and B and discard profiled statistics in the tail where only one application is running. The timing relationship among A, B and the wrapper script is shown in the timing diagram below.

![alt text](https://raw.githubusercontent.com/UofT-EcoSystem/GPU-Virtualization-Benchmarks/master/docs/wrapper_squence.png?token=AGTJ4mhfa8Np3Abqr4S5c7lwo1Ikuy1Cks5cGBVJwA%3D%3D)



Each benchmark application should be structured as in the following code snippet

```C++
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>

#include "cuda_profiler_api.h"

volatile bool ready = false;
volatile bool should_stop = false;
const char * ready_fifo = "/tmp/ready"; 

void start_handler(int sig) {
    ready = true;
}

void stop_handler(int sig) {
    should_stop = true;
}

void do_work() {
    // Data setup...

    /* Create CUDA start & stop events to record total elapsed time of kernel execution */
    cudaEvent_t start;
    cudaError_t error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {   
        fprintf(stderr, "Failed to create start event (error code %s)!\n", 
                                                cudaGetErrorString(error));   
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {   
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", 
                                                cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    /* End event creation */ 
    
    
    /* Write to pipe to signal the wrapper script that we are done with data setup */
    char pid[10];
    sprintf(pid, "%d", getpid());

    int fd = open(ready_fifo, O_WRONLY);
    int res = write(fd, pid, strlen(pid));
    close(fd);

    if (res > 0) printf("Parboil spmv write success to the pipe!\n");
    /* End pipe writing */
    
    
    /* Spin until master tells me to start kernels */
    while (!ready);
    /* End spinning */
    
    
    /* Record the start event and start nvprof profiling */
    cudaProfilerStart();
    
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", 
                                                cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    /* End CUDA start records */
    
    
    /* Main loop to do real work */
    while (!should_stop){ 
        // Launch kernels asynchrously in a loop
        kerenl<<<grid_size, block_size>>>(...);    
    }
    /* End kernel launching */
    
    /* Record and wait for the stop event */
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", 
                                                cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    cudaThreadSynchronize();

    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", 
                                                            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    cudaProfilerStop();
    
    /* End stop CUDA event handling */
    
    
    /* Output total elapsed time */
    float msecTotal = 0.0f;
    // !!! Important: must use this print format, the data processing 
    // script requires this information 
    error = cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Total elapsed time: %f ms\n", msecTotal);
    /* End elpased time recording */


    // Clean up your crap...
}

int main(int argc, char** argv) {
    if (signal(SIGUSR1, start_handler) < 0)
        perror("Signal error");

    if (signal(SIGUSR2, stop_handler) < 0)
        perror("Signal error");

    // Do some useful setup...
}

```
