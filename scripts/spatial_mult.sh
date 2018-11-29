#!/bin/bash

source path.sh

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
    sleep 2

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
    sleep 2
    kill -SIGUSR2 $pid

    tail --pid=$pid -f /dev/null 

    # rename nvprof output
    sleep 0.5
    if [[ -f "$1"/$pid.csv ]]; then
        echo $pid
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

# parse config file to get test sets

#testcase_no, device_name, exec1_name, exec2_name, keyword_exec1, keyword_exec2
sed 1d tests.config | while IFS=, read -r test_no device exec1 exec2 key1 key2 
do

    echo first app: $exec1
    echo second app: $exec2
    #select executable
    case $exec1 in
        "cutlass_sgemm_2048")
            exec1_path=$cutlass_sgemm_2048
            ;;
        "cutlass_sgemm_1024")
            exec1_path=$cutlass_sgemm_1024
            ;;
        "parboil_spmv")
            exec1_path=$parboil_spmv
            ;;
        "tensor_gemm")
            exec1_path=$tensor_gemm
            ;;
        *)
            echo "can't find exec1!"
            exit
     esac

     case $exec2 in
        "cutlass_sgemm_2048")
            exec2_path=$cutlass_sgemm_2048
            ;;
        "cutlass_sgemm_1024")
            exec2_path=$cutlass_sgemm_1024
            ;;
        "parboil_spmv")
            exec2_path=$parboil_spmv
            ;;
        "tensor_gemm")
            exec2_path=$tensor_gemm
            ;;
        *)
            echo "can't find exec2!"
            exit
     esac



    # Multiple runs to get the most stable run
    for j in `seq 0 8`; do
        echo Iteration $j


        filepath=experiments/$test_no/run$j

        rm -rf $filepath
        mkdir -p $filepath

        echo $filepath

        # nvprof all processes
        $NVPROF --profile-from-start off --profile-all-processes -f --print-gpu-trace --csv --normalized-time-unit ms --log-file $filepath/%p.csv &
        #$NVPROF --profile-from-start off --profile-all-processes -f -o $filepath/%p.prof &
        pid_nvp=$( echo $! )

        sleep 2
        echo Isolated runs

        echo dummy: "${exec1_path}"


        run_single $filepath "${exec1_path}" dummy

        echo 1: "${exec1_path}"
        run_single $filepath "${exec1_path}" single-1
        echo 2: "${exec2_path}"
        run_single $filepath "${exec2_path}" single-2

        echo Time Multiplexing 

        run_concurrent $filepath "${exec1_path}" time-1 "${exec2_path}" time-2

        echo MPS run

        source common/mps_server_on.sh

        run_concurrent $filepath "${exec1_path}" mps-1 "${exec2_path}" mps-2


        source common/mps_server_off.sh

        # need to cancel nvprof and change file path
        kill -SIGINT $pid_nvp

        wait

        echo Done iteration $j
    done


    # change ownership of the experiment folder to yourself
    chown -R $usrname: experiments/$test_no



done




