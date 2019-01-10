#!/bin/bash

source ../config
source run_path.sh
source nv-metrics

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
    echo "get pipe"

    kill -SIGUSR1 $pid
    sleep 1
    kill -SIGUSR2 $pid

    echo "wait for the app to finish"

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

run_single_metric() {
    # usage: run_single_metric filepath app app_out
    echo "$2"
    $NVPROF --profile-from-start off -f --csv --log-file "$1"/"$3".csv -m $metrics $2 &

    local pid=$( cat $pipe )

    kill -SIGUSR1 $pid

    sleep 1

    kill -SIGUSR2 $pid

    wait

}

run_single_nvvp() {
    # usage: run_single_metric filepath app app_out
    echo "$2"
    $NVPROF --profile-from-start off -f -o "$1"/"$3".prof --analysis-metrics $2 &

    local pid=$( cat $pipe )

    kill -SIGUSR1 $pid

    sleep 1

    kill -SIGUSR2 $pid

    wait

}

run_single_inst_count() {
    # usage: run_single_inst_count filepath app app_out
    $NVPROF --profile-from-start off --csv --log-file "$1"/"$3"_inst.csv -m inst_executed $2 &

    local pid=$( cat $pipe )

    kill -SIGUSR1 $pid

    sleep 1

    kill -SIGUSR2 $pid

    wait

}

# Start of script
if [ "$#" -ne 3 ]; then
        echo "Illegal number of parameters"
        echo "Usage: <path/to/script> <timeline | duration | metrics | nvvp> <# of iters> <tests.config>"
        exit
fi

export TOP_PID=$$

echo $TOP_PID

trap "rm -f $pipe" EXIT

if [[ ! -f $pipe ]]; then
    mkfifo $pipe
fi

# parse config file to get test sets

#testcase_no, device_name, exec1_name, exec2_name
sed 1d $3 | while IFS=, read -r test_no device exec1 exec2 key1 key2
do

    #select executable
    echo exec1: $exec1
    exec1_path=$(select_run $exec1)

    if [ "can't find exec!" = "$exec1_path" ]; then
        echo Cannot find exec1. Exiting...
        exit 1
    fi

    echo First app: "$exec1_path"


    echo exec2: $exec2
    exec2_path=$(select_run $exec2)

    if [ "can't find exec!" = "$exec2_path" ]; then
        echo Cannot find exec2. Exiting...
        exit 1
    fi

    echo Second app: "$exec2_path"



    let "iter = $2 - 1"
    # Multiple runs to get the most stable run
    for j in `seq 0 $iter`; do
        echo Iteration $j


        filepath=$ROOT/tests/experiments/$test_no/run$j

        rm -rf $filepath
        mkdir -p $filepath

        echo $filepath

        # nvprof all processes
        case $1 in 
            "timeline")
                echo ">>>>>>>>>>>> Capture application timeline <<<<<<<<<<<<<<"

                concurrent=true

                $NVPROF --profile-from-start off --profile-all-processes -f -o $filepath/%p.prof &


                ;;
            "duration")
                echo ">>>>>>>>>>>> Capture application runtime <<<<<<<<<<<<<<"

                concurrent=true

                # To calculate IPC, we need to get number of instructions executed by each kernel
                # first. Only do this for the first run.
                if [ $j -eq 0 ]; then
                    run_single_inst_count $filepath "${exec1_path}" single-1
                    run_single_inst_count $filepath "${exec2_path}" single-2
                fi

                # Now, fire the actual nvprof instance to capture runtime duration

                $NVPROF --profile-from-start off --profile-all-processes -f --print-gpu-trace --csv --normalized-time-unit ms --log-file $filepath/%p.csv &

                ;;

            "metrics")
                echo ">>>>>>>>>>>> Capture application perf counters <<<<<<<<<<<<<<"

                run_single_metric $filepath "${exec1_path}" single-1
                run_single_metric $filepath "${exec2_path}" single-2

                concurrent=false

                ;;
            "nvvp")
                echo ">>>>>>>>>>>> Capture application nvvp analysis <<<<<<<<<<<<<<"

                run_single_nvvp $filepath "${exec1_path}" single-1
                run_single_nvvp $filepath "${exec2_path}" single-2

                concurrent=false

                ;;


            *)
                echo ">>>>>>>>>>> Option unknown! Exiting... <<<<<<<<<<<<<<<<"
                exit
                ;;
        esac

        if [ "$concurrent" = true ]; then
            pid_nvp=$( echo $! )

            sleep 2
            echo Isolated runs

            echo 1: "${exec1_path}"
            run_single $filepath "${exec1_path}" single-1
            echo 2: "${exec2_path}"
            run_single $filepath "${exec2_path}" single-2


            #echo Time Multiplexing 

            #run_concurrent $filepath "${exec1_path}" time-1 "${exec2_path}" time-2

            echo MPS run

            source ../mps/mps_server_on.sh

            run_concurrent $filepath "${exec1_path}" mps-1 "${exec2_path}" mps-2

            source ../mps/mps_server_off.sh

            # need to cancel nvprof and change file path
            kill -SIGINT $pid_nvp

        fi

        wait

        echo Done iteration $j
    done


    # change ownership of the experiment folder to yourself
    chown -R $usrname: $ROOT/tests/experiments/$test_no

done




