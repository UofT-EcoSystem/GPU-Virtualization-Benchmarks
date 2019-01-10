# Benchmarks for GPU Virtualization 
This repo is a suite of GPGPU benchmarks that will be used to collect both motivation data for GPU resource virtualization and to evaluate our proposed solutions. 


## List of Benchmark Suites

1. [Parboil](http://impact.crhc.illinois.edu/parboil/parboil.aspx)
* bfs
* cutcp
* histo
* lbm
* mri-gridding
* mri-q
* sad
* sgemm
* spmv
* stencil
* tpacf

2. [Cutlass](https://github.com/NVIDIA/cutlass)
* sgemm
* wmma

3. [cuda-sdk](https://docs.nvidia.com/cuda/cuda-samples/index.html)
* tensorTensorCoreGemm

4. [TBD](http://tbd-suite.ai/)
* ImageClassification-Inception_v3
* MachineTranslation-Seq2Seq
   

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

Concurrent execution type of options will invoke isolated run (one application at a time), 
time multiplexed run (all applications within the same test simultaneously using time-sliced 
scheduler on the hardware)
and [MPS](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf) run 
(similar to time multiplexed run but using MPS' spatial multiplexing scheduler).

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
The test config describes what benchmarks you would like to run in parallel for each test. 
An example config looks like the following:

```
testcase_no, device_name, exec1_name, exec2_name, keyword_exec1, keyword_exec2
f7,Volta,cutlass_sgemm_512,cutlass_sgemm_2048,spmv,Sgemm
```
The first line is always the same headers. Each of the following line is a new test case.

`testcase_no` should be a unique identifier of the test which will be the name of the generated
experiment result folder under `$PROJECT_ROOT/tests/experiment`. 

`device name` is a user friendly
description of your device used by the postprocessing scripts for plot titles. 

`exec1_name` and `exec2_name` are the benchmark name defined 
in `$PROJECT_ROOT/scripts/run/run_path.sh`. 

`keyword_exec1` and `keyword_exec2` are user-friendly description of the benchmarks for 
postprocessing scripts. 

## Experiment Results Post-processing

All profiled results generated will be stored under `PROJECT_ROOT/tests/experiment`. A set of
postproecssing scripts written using Python Matplotlib are available for your convenience. They
are located in `PROJECT_ROOT/scripts/process-data`.

#### Existing Processing Scripts





## To Add New Benchmark Source

#### Interface Set up in Source

To use our benchmark wrapper script, each benchmark should implement the same interface to 
synchronize kernel launches. Consider a test comprises two applications A and B, with each 
application invoking different GPU compute kernels. The goal is to capture performance data 
during the period where both kernels from A and B execute concurrently. Hence, kernel execution 
in each application should start at the same time and ideally end around the same time. Since 
it's impossible to guarantee all kernels end simultaneously due to kernel runtime difference, 
the data processing script will calculate the delta of kernel elapased time between A and B 
and discard profiled statistics in the tail where only one application is running. The timing 
relationship among A, B and the wrapper script is shown in the timing diagram below.

![alt text](https://raw.githubusercontent.com/UofT-EcoSystem/GPU-Virtualization-Benchmarks/master/docs/wrapper_squence.png?token=AGTJ4mhfa8Np3Abqr4S5c7lwo1Ikuy1Cks5cGBVJwA%3D%3D)


Each benchmark application should be structured similar to the code snippet 
in `$PROJECT_ROOT/benchmarks/interface.c`

#### Build System Set up ####

Your Makefile or CMakeLists should source/include `$PROJECT_ROOT/scripts/config` for build 
information such as CUDA path, CUDA device compute capability etc.. 

Also, modify the script `$PROJECT_ROOT/scripts/compile/compile.sh` to add support 
for compilation through the main driver script `virbench.sh`.

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


