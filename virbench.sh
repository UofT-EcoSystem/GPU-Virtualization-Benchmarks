#!/bin/bash

print_usage() {
    echo "Usage:"
    echo "To build: virbench.sh <compile | clean> <benchmark suite name>"
    echo "To run: virbench.sh run <timeline | duration | metrics | nvvp> 
          <# of iters> <tests.config>"
}


case $1 in

"compile" | "clean")
    if [ "$#" -ne 2 ]; then
        print_usage
        exit
    fi

    (cd scripts/compile && ./compile.sh $1 $2)

    ;;
"run")
    if [ "$#" -ne 4 ]; then
        print_usage
        exit
    fi

    case $4 in
         /*) test_config=$4;;
         *) test_config=$(pwd)/$4;;
    esac

    (cd scripts/run && ./run.sh $2 $3 $test_config)


    ;;
*)
    print_usage
    exit

    ;;


esac
