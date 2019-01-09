source ../config


CUTLASS_PATH="$ROOT/benchmarks/cutlass/build"

PARBOIL_PATH="$ROOT/benchmarks/parboil"

TENSOR_PATH="$ROOT/benchmarks/cuda-sdk/cudaTensorCoreGemm"

parboil_spmv="$PARBOIL_PATH/benchmarks/spmv/build/cuda_default/spmv -i /home/serinatan/gpu-sharing-exp/parboil/datasets/spmv/large/input/Dubcova3.mtx.bin,/home/serinatan/gpu-sharing-exp/parboil/datasets/spmv/large/input/vector.bin -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/spmv/run/large/Dubcova3.mtx.out" 

parboil_sgemm="$PARBOIL_PATH/benchmarks/sgemm/build/cuda_default/sgemm -i /home/serinatan/gpu-sharing-exp/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/serinatan/gpu-sharing-exp/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/serinatan/gpu-sharing-exp/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/sgemm/run/medium/matrix3.txt"

parboil_mri_gridding="$PARBOIL_PATH/benchmarks/mri-gridding/build/cuda_default/mri-gridding -i /home/serinatan/gpu-sharing-exp/parboil/datasets/mri-gridding/small/input/small.uks -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/mri-gridding/run/small/output.txt -- 32 0"

parboil_stencil="$PARBOIL_PATH/benchmarks/stencil/build/cuda_default/stencil -i /home/serinatan/gpu-sharing-exp/parboil/datasets/stencil/default/input/512x512x64x100.bin -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/stencil/run/default/512x512x64.out -- 512 512 64 100 "

parboil_cutcp="$PARBOIL_PATH/benchmarks/cutcp/build/cuda_default/cutcp -i /home/serinatan/gpu-sharing-exp/parboil/datasets/cutcp/large/input/watbox.sl100.pqr -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/cutcp/run/large/lattice.dat"

parboil_histo="$PARBOIL_PATH/benchmarks/histo/build/cuda_default/histo -i /home/serinatan/gpu-sharing-exp/parboil/datasets/histo/large/input/img.bin -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/histo/run/large/ref.bmp -- 10000 4"

parboil_sad="$PARBOIL_PATH/benchmarks/sad/build/cuda_default/sad -i /home/serinatan/gpu-sharing-exp/parboil/datasets/sad/large/input/reference.bin,/home/serinatan/gpu-sharing-exp/parboil/datasets/sad/large/input/frame.bin -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/sad/run/large/out.bin"

parboil_lbm="$PARBOIL_PATH/benchmarks/lbm/build/cuda_default/lbm -i /home/serinatan/gpu-sharing-exp/parboil/datasets/lbm/long/input/120_120_150_ldc.of -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/lbm/run/long/reference.dat -- 1000"

parboil_mriq="$PARBOIL_PATH/benchmarks/mri-q/build/cuda_default/mri-q -i /home/serinatan/gpu-sharing-exp/parboil/datasets/mri-q/large/input/64_64_64_dataset.bin -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/mri-q/run/large/64_64_64_dataset.out"

parboil_tpacf="$PARBOIL_PATH/benchmarks/tpacf/build/cuda_default/tpacf -i /home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Datapnts.1,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.1,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.2,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.3,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.4,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.5,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.6,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.7,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.8,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.9,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.10,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.11,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.12,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.13,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.14,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.15,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.16,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.17,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.18,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.19,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.20,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.21,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.22,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.23,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.24,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.25,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.26,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.27,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.28,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.29,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.30,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.31,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.32,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.33,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.34,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.35,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.36,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.37,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.38,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.39,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.40,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.41,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.42,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.43,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.44,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.45,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.46,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.47,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.48,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.49,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.50,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.51,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.52,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.53,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.54,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.55,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.56,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.57,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.58,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.59,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.60,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.61,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.62,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.63,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.64,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.65,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.66,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.67,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.68,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.69,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.70,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.71,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.72,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.73,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.74,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.75,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.76,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.77,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.78,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.79,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.80,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.81,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.82,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.83,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.84,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.85,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.86,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.87,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.88,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.89,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.90,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.91,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.92,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.93,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.94,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.95,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.96,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.97,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.98,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.99,/home/serinatan/gpu-sharing-exp/parboil/datasets/tpacf/large/input/Randompnts.100 -o /home/serinatan/gpu-sharing-exp/parboil/benchmarks/tpacf/run/large/tpacf.out -- -n 100 -p 10391"


cutlass_sgemm_256="$CUTLASS_PATH/examples/00_basic_gemm/00_basic_gemm 256 256 256"
cutlass_sgemm_512="$CUTLASS_PATH/examples/00_basic_gemm/00_basic_gemm 512 512 512"
cutlass_sgemm_1024="$CUTLASS_PATH/examples/00_basic_gemm/00_basic_gemm 1024 1024 1024"
cutlass_sgemm_2048="$CUTLASS_PATH/examples/00_basic_gemm/00_basic_gemm 2048 2048 2048"
cutlass_sgemm_4096="$CUTLASS_PATH/examples/00_basic_gemm/00_basic_gemm 4096 4096 4096"

cutlass_igemm_256="$CUTLASS_PATH/examples/00_basic_gemm_int/00_basic_gemm_int 256 256 256"
cutlass_igemm_512="$CUTLASS_PATH/examples/00_basic_gemm_int/00_basic_gemm_int 512 512 512"
cutlass_igemm_1024="$CUTLASS_PATH/examples/00_basic_gemm_int/00_basic_gemm_int 1024 1024 1024"
cutlass_igemm_2048="$CUTLASS_PATH/examples/00_basic_gemm_int/00_basic_gemm_int 2048 2048 2048"
cutlass_igemm_4096="$CUTLASS_PATH/examples/00_basic_gemm_int/00_basic_gemm_int 4096 4096 4096"

cutlass_wmma="$CUTLASS_PATH/tools/test/perf/cutlass_perf_test --kernels=wmma_gemm_nn"


tensor_gemm="$TENSOR_PATH/cudaTensorCoreGemm"


select_run() {
    case $1 in
        "cutlass_wmma")
            local exec_path=$cutlass_wmma
            ;;
        "cutlass_igemm_256")
            local exec_path=$cutlass_igemm_256
            ;;
        "cutlass_igemm_512")
            local exec_path=$cutlass_igemm_512
            ;;
        "cutlass_igemm_1024")
            local exec_path=$cutlass_igemm_1024
            ;;
        "cutlass_igemm_2048")
            local exec_path=$cutlass_igemm_2048
            ;;
        "cutlass_igemm_4096")
            local exec_path=$cutlass_igemm_4096
            ;;
        "cutlass_sgemm_256")
            local exec_path=$cutlass_sgemm_256
            ;;
        "cutlass_sgemm_512")
            local exec_path=$cutlass_sgemm_512
            ;;
        "cutlass_sgemm_1024")
            local exec_path=$cutlass_sgemm_1024
            ;;
        "cutlass_sgemm_2048")
            local exec_path=$cutlass_sgemm_2048
            ;;
        "cutlass_sgemm_4096")
            local exec_path=$cutlass_sgemm_4096
            ;;
        "parboil_spmv")
            local exec_path=$parboil_spmv
            ;;
        "parboil_histo")
            local exec_path=$parboil_histo
            ;;
        "parboil_mriq")
            local exec_path=$parboil_mriq
            ;;
        "parboil_mri_gridding")
            local exec_path=$parboil_mri_gridding
            ;;
        "parboil_cutcp")
            local exec_path=$parboil_cutcp
            ;;
        "parboil_sad")
            local exec_path=$parboil_sad
            ;;
        "parboil_stencil")
            local exec_path=$parboil_stencil
            ;;
        "parboil_lbm")
            local exec_path=$parboil_lbm
            ;;
        "parboil_tpacf")
            local exec_path=$parboil_tpacf
            ;;
        "parboil_sgemm")
            local exec_path=$parboil_sgemm
            ;;
        "tensor_gemm")
            local exec_path=$tensor_gemm
            ;;

        *)
            echo "can't find exec1!"
            exit
     esac

     echo "$exec_path"
 }

