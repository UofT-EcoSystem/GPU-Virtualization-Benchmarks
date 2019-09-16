#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "Illegal number of parameters"
  echo "Usage: run.sh <benchmark[-benchmark...]> <sim|hw> [gdb|nsight|nvprof]"
  echo "nsight/nvprof is only available for hw mode."
fi


BENCH_ROOT="$(dirname "$(pwd)")"
PARBOIL_ROOT="$BENCH_ROOT/parboil"
PARBOIL_DATA=$PARBOIL_ROOT/datasets
CUTLASS_INPUT="$BENCH_ROOT/cutlass/input"
NSIGHT_SECTION=$BENCH_ROOT/../sections
gpgpusim=/mnt/ecosystem-gpgpu-sim/

# define benchmark to dataset pair (largest available set)
#declare -A datamap=(["cutcp"]="large" ["histo"]="large" ["lbm"]="long" ["mri_gridding"]="small" \


#declare -A datamap=(["cutcp"]="large" ["sgemm"]="medium" ["tpacf"]="large" ["lbm"]="long" ["sad"]="large" ["spmv"]="large" ["stencil"]="default") 
declare -A datamap=(["parb_sgemm"]="$PARBOIL_DATA/sgemm/medium/input" \
                    ["parb_stencil"]="$PARBOIL_DATA/stencil/default/input" \
                    ["parb_lbm"]="$PARBOIL_DATA/lbm/short/input" \
                    ["cut_sgemm"]="$CUTLASS_INPUT/sgemm" \
                    ["cut_wmma"]="$CUTLASS_INPUT/wmma")

PROFILE=""


if [ "$2" == "sim" ]; then
  PARBOIL_BUILD=$PARBOIL_ROOT/build-docker
  if [ "$3" == "gdb" ]; then
    PROFILE="gdb --args "
  fi
 
else
  PARBOIL_BUILD=$PARBOIL_ROOT/build

  if [ "$3" == "nsight" ]; then
    #PROFILE="nv-nsight-cu-cli -f --csv --section-folder $NSIGHT_SECTION --section Memory_Usage "
    PROFILE="sudo nv-nsight-cu-cli -f --csv "
  elif [ "$3" == "nvprof" ]; then
    PROFILE="sudo /usr/local/cuda/bin/nvprof -f --print-gpu-trace --csv "
  fi
fi



#function param: $1=benchmark
function build_input() {
  folder="${datamap[$1]}"

  IO="$1 "

  # grab input params from description file
  if [ -f "$folder/DESCRIPTION" ]; then
    while IFS='\n' read -r line; do
      # split key values
      IFS=':' read -ra words <<< "$line"

      if [ "${words[0]}" == "Inputs" ]; then
        IO="$IO-i "
        # split inputs
        IFS=' ' read -ra inputs <<< "${words[1]}"

        empty=true
        for i in ${inputs[@]}; do 
          if [ "$empty" = false ]; then
            IO="$IO,"
          fi

          IO="$IO$folder/$i"

          empty=false
        done

        IO="$IO -o $PARBOIL_BUILD/out/$1/$1.out"
      fi


      if [ "${words[0]}" == "Parameters" ]; then
        IO="$IO ${words[1]}"
      fi

    done < "$folder/DESCRIPTION"

  else
    #IO="$IO-i $folder/$(ls $folder) -o $PARBOIL_BUILD/out/$1/$1.out"
    echo "No input description file."
    exit
  fi


  mkdir -p $PARBOIL_BUILD/out/$1
  
  echo "$IO"
}

# make results folder if none exists
#mkdir -p results

IFS='-' read -ra benchmarks <<< "$1"

# make a new build folder for this run
mkdir -p $BENCH_ROOT/build
rm -rf $BENCH_ROOT/build/$1
mkdir -p $BENCH_ROOT/build/$1

inputs=""
defines=""
for bench in ${benchmarks[@]}; do
  if [[ -v "datamap[$bench]" ]]; then
    echo ">>>>>>>>> Running kernel: $bench <<<<<<<<<"
  else
    echo "Benchmarks do not exist: $bench"
    exit
  fi

  # make input files
  IO=$(build_input "$bench")
  txt=$BENCH_ROOT/build/$1/$bench.txt
  echo "$IO" > $txt

  inputs="$inputs$txt "
  defines="$defines-D$bench=ON "
done

# compile a new build!
cd $BENCH_ROOT/build/$1 && cmake $defines $BENCH_ROOT && make -j all VERBOSE=1

if [ "$2" == "sim" ]; then
  # copy gpgpu-sim config
  cp $BENCH_ROOT/scripts/gpgpusim.config .
  cp $BENCH_ROOT/scripts/config_volta_islip.icnt .

  # source gpgpusim setup scripts
  cd $gpgpusim && source setup_environment
fi

mkdir -p $BENCH_ROOT/results

if [ ${#benchmarks[@]} -eq 1 ]; then
  mkdir -p $BENCH_ROOT/results/seq
  res_folder=$BENCH_ROOT/results/seq
else
  mkdir -p $BENCH_ROOT/results/conc
  res_folder=$BENCH_ROOT/results/conc
fi

cd $BENCH_ROOT/build/$1
# copy the so file to this folder
source ../../scripts/set_lib.sh lib/ rel
ldd driver
# run the app
$PROFILE ./driver $inputs | tee $res_folder/$1.txt




