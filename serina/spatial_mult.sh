#!/bin/bash


usrname=serinatan
NVPROF="/usr/local/cuda-9.0/bin/nvprof"

CUTLASS_PATH="/home/serinatan/gpu-sharing-exp/nvidia/samples/cutlass/build/examples/00_basic_gemm"

PARBOIL_PATH="/home/serinatan/gpu-sharing-exp/parboil"

TENSOR_PATH="/home/serinatan/gpu-sharing-exp/nvidia/samples/cudaTensorCoreGemm"

parboil_run="$PARBOIL_PATH/benchmarks/spmv/build/cuda_default/spmv -i /home/serinatan/gpu-sharing-exp/parboil/datasets/spmv/large/input/Dubcova3.mtx.bin,/home/serinatan/gpu-sharing-exp/parboil/datasets/spmv/large/input/vector.bin -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/spmv/run/large/Dubcova3.mtx.out" 

cutlass_run="$CUTLASS_PATH/00_basic_gemm"

tensor_run="$TENSOR_PATH/cudaTensorCoreGemm"


pipe=/tmp/ready

run_concurrent() {
    # usage: run_concurrent filepath app1 app1_out app2 app2_out
    eval "$2 > $1/$3.txt &"
    local pid1=$( echo $! )

    eval "$4 > $1/$5.txt &"
    local pid2=$( echo $!)

    for line in `seq 0 1`; do
        cat $pipe
    done
    
    # start kernels at the same time
    kill -SIGUSR1 $pid1
    kill -SIGUSR1 $pid2

    # let the kernels run for 2 sec
    sleep 1

    # stop kernels at the same time
    kill -SIGUSR2 $pid1 &
    kill -SIGUSR2 $pid2


    tail --pid=$pid1 -f /dev/null 
    tail --pid=$pid2 -f /dev/null 

    # rename nvprof output
    sleep 0.5
    if [[ -f "$1"/$pid1.csv ]]; then
        mv "$1"/$pid1.csv "$1"/"$3".csv
    fi

    if [[ -f "$1"/$pid2.csv ]]; then
        mv "$1"/$pid2.csv "$1"/"$5".csv
    fi

    if [[ -f "$1"/$pid1.prof ]]; then
        mv "$1"/$pid1.prof "$1"/"$3".prof
    fi

    if [[ -f "$1"/$pid2.prof ]]; then
        mv "$1"/$pid2.prof "$1"/"$5".prof
    fi

}

run_single() {
    # usage: run_single filepath app1 app1_out
    eval "$2 > $1/$3.txt &"
    local pid=$( echo $! )

    cat $pipe
    kill -SIGUSR1 $pid
    sleep 1
    kill -SIGUSR2 $pid

    tail --pid=$pid -f /dev/null 

    # rename nvprof output
    sleep 0.5
    if [[ -f "$1"/$pid.csv ]]; then
        mv "$1"/$pid.csv "$1"/"$3".csv
    fi

    if [[ -f "$1"/$pid.prof ]]; then
        mv "$1"/$pid.prof "$1"/"$3".prof
    fi

}

# Start of script


trap "rm -f $pipe" EXIT

if [[ ! -p $pipe ]]; then
    mkfifo $pipe
fi

# Multiple runs to get the most stable run
for j in `seq 0 9`; do
    echo Iteration $j


    filepath=experiments/set200/run$j

    rm -rf $filepath
    mkdir -p $filepath

    echo $filepath

    # nvprof all processes
    $NVPROF --profile-from-start off --profile-all-processes -f --print-gpu-trace --csv --normalized-time-unit ms --log-file $filepath/%p.csv &
    #$NVPROF --profile-from-start off --profile-all-processes -f -o $filepath/%p.prof &
    pid_nvp=$( echo $! )

    sleep 2
    echo Isolated runs

    #run_single $filepath "${parboil_run}" single-1
    run_single $filepath "${tensor_run}" single-1
    run_single $filepath "${cutlass_run}" single-2

    echo Time Multiplexing 

    #run_concurrent $filepath "${parboil_run}" time-1 "${cutlass_run}" time-2
    run_concurrent $filepath "${tensor_run}" time-1 "${cutlass_run}" time-2

    echo MPS run

    source common/mps_server_on.sh

    #run_concurrent $filepath "${parboil_run}" mps-1 "${cutlass_run}" mps-2
    run_concurrent $filepath "${tensor_run}" mps-1 "${cutlass_run}" mps-2


    source common/mps_server_off.sh

    # need to cancel nvprof and change file path
    kill -SIGINT $pid_nvp

    wait

    echo Done iteration $j
done


# change ownership of the experiment folder to yourself
chown -R $usrname: experiments/set200
