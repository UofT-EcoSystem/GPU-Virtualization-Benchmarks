# Benchmarks for GPU Virtualization 
This repo is a suite of GPGPU benchmarks that will be used to collect both motivation data for GPU resource virtualization and to evaluate our proposed solutions. 


## List of Benchmark Suites

1. [Parboil](http://impact.crhc.illinois.edu/parboil/parboil.aspx)
..* bfs
..* cutcp
..* histo
..* lbm
..* mri-gridding
..* mri-q
..* sad
..* sgemm
..* spmv
..* stencil
..* tpacf

2. [Cutlass](https://github.com/NVIDIA/cutlass)
..* sgemm
..* wmma

3. [cuda-sdk](https://docs.nvidia.com/cuda/cuda-samples/index.html)
..* tensorTensorCoreGemm

4. [TBD](http://tbd-suite.ai/)
..* ImageClassification-Inception_v3
..* MachineTranslation-Seq2Seq
   

## To Build Benchmarks

#### Modify config files

You can modify the following parameters in config file `PROJECT_ROOT/scripts/run/path.sh`


* `usrname`: Your own username. Some of your tests might need to run with sudo rights and hence the experiment result folders will be sudo owned by default. The run scripts will automatically change ownership of these folders to you at the end of the test run.
* `CUDAHOME`: Path to CUDA root folder. 
* `Arch version`: SM version of your CUDA device.

#### Set up datasets of Parboil
1. Download and extract datasets to `PARBOIL_ROOT/datasets`
2. To compile all benchmarks: `source PARBOIL_ROOT/compile.sh`
3. To clean all benchmarks: `source PARBOIL_ROOT/clean.sh`


#### Compile/Clean benchmarks

Run `$PROJECT_ROOT/virbench.sh <compile | clean> <parboil | cutlass | cuda-sdk>` to compile/clean 
projects. The script will build ALL benchmarks within the benchmark suite. 


## To Run Benchmarks

Run `$PROJECT_ROOT/virbench.sh run <timeline | duration | metrics | nvvp> <# of iterations> <test.config>` 
to run benchmarks. The run options and test.config and explained in the following sections. 
The `# of iterations` indicates how many times the tests defined in test.config are run, which 
is useful when multiple runs are required for data processing. 

#### Run Script Options

_Concurent execution:_
1. **timeline**: Capture execution timeline of the benchmark only. Output is a .prof file that 
can be imported by Nvidia's visual profiler (nvvp).

2. **duration**: Capture runtime duration of each kernel within the benchmark. Output is a .csv file.

_Isolated execution:_
3. **metrics**: Evaluate each kernel within the benchmark in terms of a predefined metrics 
supported by nvprof. The default metrics are listed in [NVPROF Metrics Considered](#metrics).
The metrics evaluated by the script can be modified in `PROJECT_ROOT/scripts/run/metrics` file.
You may run `CUDAHOME/bin/nvprof --query-metrics` on Volta and earlier devices to find 
a complete list of metrics supported by nvprof. The metrics option is not available for Turing 
and newer devices due to deprecation of nvprof (future work for support of new profile tools). 
Output is a .csv file.

4. **nvvp**: Capture both timeline and all analysis metric statistics of the benchmark. The output
is a .prof file that can be imported by nvvp. With the analysis metric information, nvvp will also
provide detailed performance analysis (memory/compute-bound etc.) and optimization suggestions on 
the benchmark.

#### Write Test Configs


## Experiment Results Post-processing
Overview

#### Existing Processing Scripts





## To Add New Benchmark Source

#### Interface Set up in Source

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

#### Build System Set up ####

blah...


## NVPROF Metrics Considered (#metrics)

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


