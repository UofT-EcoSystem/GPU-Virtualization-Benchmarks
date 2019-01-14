#!/bin/bash

source ../config

export CUDA_BIN_PATH=$CUDAHOME

case $1 in 
    "clean")
        case $2 in 
            "parboil")
                cd $ROOT/benchmarks/parboil
                source clean.sh
                ;;
                
            "cutlass")
                cd $ROOT/benchmarks/cutlass
                if [ ! -d "build/" ]; then
                    mkdir build && cd build 
                    cmake -CUTLASS_NVCC_ARCHS=$cmake_arch
                    cmake ..
                else
                    cd build
                fi

                make clean

                ;;
            "cuda-sdk")
                cd $ROOT/benchmarks/cuda-sdk

                for dir in ./*
                do
                    cd $dir
                    make clean
                    cd ..
                done
                ;;
            *)
                echo ">>>>>>>>>>>>>>>>> Benchmark unknown! Exiting... <<<<<<<<<<<<<<<<"
                echo "Usage: ./build [compile | clean] [parboil | cutlass | cuda-sdk]"
                exit
                ;;
        esac
 
        ;;
    "compile")
        case $2 in 
            "parboil")
                cd $ROOT/benchmarks/parboil
                source compile.sh
                ;;
                
            "cutlass")
                cd $ROOT/benchmarks/cutlass
                if [ ! -d "build/" ]; then
                    mkdir build && cd build 
                    cmake ..
                else
                    cd build
                fi

                make -j12

                ;;
            "cuda-sdk")
                cd $ROOT/benchmarks/cuda-sdk

                for dir in ./*
                do
                    echo $dir
                    cd $dir
                    make -j12
                    cd ..
                done
                ;;
            *)
                echo ">>>>>>>>>>>>>>>>> Benchmark unknown! Exiting... <<<<<<<<<<<<<<<<"
                echo "Usage: ./build [compile | clean] [parboil | cutlass | cuda-sdk]"
                exit
                ;;
        esac
        ;;
    *)
        echo ">>>>>>>>>>>>>>> Option unknown! Exiting... <<<<<<<<<<<<<<"
        echo "Usage: ./build [compile | clean] [parboil | cutlass | cuda-sdk]"
        exit
        ;;
esac




         
