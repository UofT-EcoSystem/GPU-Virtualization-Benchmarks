# Need to run experiments in sudo mode... 
# change the ownership experiment result folders back to yourself
usrname=serinatan

# Path to nvprof
NVPROF="/usr/local/cuda-9.0/bin/nvprof"

# Shared named pipe for inter-application communication
pipe=/tmp/ready

# Path to benchmark sources
CUTLASS_PATH="/home/serinatan/gpu-sharing-exp/nvidia/samples/cutlass/build/examples/00_basic_gemm"
PARBOIL_PATH="/home/serinatan/gpu-sharing-exp/parboil"
TENSOR_PATH="/home/serinatan/gpu-sharing-exp/nvidia/samples/cudaTensorCoreGemm"

# Executables
parboil_spmv="$PARBOIL_PATH/benchmarks/spmv/build/cuda_default/spmv -i /home/serinatan/gpu-sharing-exp/parboil/datasets/spmv/large/input/Dubcova3.mtx.bin,/home/serinatan/gpu-sharing-exp/parboil/datasets/spmv/large/input/vector.bin -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/spmv/run/large/Dubcova3.mtx.out" 

cutlass_sgemm_2048="$CUTLASS_PATH/00_basic_gemm 2048 2048 2048"
cutlass_sgemm_1024="$CUTLASS_PATH/00_basic_gemm 1024 1024 1024"

tensor_gemm="$TENSOR_PATH/cudaTensorCoreGemm"


