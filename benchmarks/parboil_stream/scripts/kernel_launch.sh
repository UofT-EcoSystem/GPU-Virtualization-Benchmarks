#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "Illegal number of parameters"
  echo "Usage: run.sh <benchmark> <sim|hw> [nsight|nvprof]"
  echo "nsight/nvprof is only available for hw mode."
fi


PARBOIL_ROOT="$(dirname "$(pwd)")"
PARBOIL_DATA=$PARBOIL_ROOT/datasets
NSIGHT_SECTION=$PARBOIL_ROOT/../sections
gpgpusim=/mnt/ecosystem-gpgpu-sim/

# define benchmark to dataset pair (largest available set)
#declare -A datamap=(["cutcp"]="large" ["histo"]="large" ["lbm"]="long" ["mri_gridding"]="small" \
#  ["mri_q"]="large" ["sad"]="large" ["sgemm"]="medium" ["spmv"]="large" ["stencil"]="default" \
#  ["tpacf"]="large")


declare -A datamap=(["cutcp"]="large" ["sgemm"]="medium" ["tpacf"]="large" ["lbm"]="long" ["sad"]="large" ["spmv"]="large" ) 

PROFILE=""


if [ "$2" == "sim" ]; then
  PARBOIL_BUILD=$PARBOIL_ROOT/build-docker
else
  PARBOIL_BUILD=$PARBOIL_ROOT/build

  if [ "$3" == "nsight" ]; then
    #PROFILE="nv-nsight-cu-cli -f --csv --section-folder $NSIGHT_SECTION --section Memory_Usage "
    PROFILE="nv-nsight-cu-cli -f --csv "
  elif [ "$3" == "nvprof" ]; then
    PROFILE="/usr/local/cuda-10.0/bin/nvprof -f -o conc.prof "
  fi
fi



#function param: $1=benchmark $2=dataset
function build_input() {
  folder=$PARBOIL_DATA/$1/$2/input

  IO=""

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

        IO="$IO -o $PARBOIL_BUILD/out/$1/$1_$2.out"
      fi


      if [ "${words[0]}" == "Parameters" ]; then
        IO="$IO ${words[1]}"
      fi

    done < "$folder/DESCRIPTION"

  else
    IO="$IO-i $folder/$(ls $folder) -o $PARBOIL_BUILD/out/$1/$1_$2.out"
  fi


  mkdir -p $PARBOIL_BUILD/out/$1
  
  echo "$IO"
}

mkdir -p results

# isolated run:

bench=$1

# compile a new build!
cd .. && mkdir -p build-$bench 
cd build-$bench && rm -rf * && cmake -D$bench=ON .. && make -j8
back=$(pwd)

if [ "$2" == "sim" ]; then
  # manually relink the executable with shared runtime
  cp ../scripts/link.sh .
  ./link.sh

  # copy gpgpu-sim config
  cp ../scripts/gpgpusim.config .
  cp ../scripts/config_fermi_islip.icnt .

  # source gpgpusim setup scripts
  cd $gpgpusim
  source setup_environment
fi

cd $back

IO=$(build_input "$bench" "${datamap[$bench]}")

echo -e "\n\n>>>>>>>>>>>>>>>> Launching kernel in isolation: $bench <<<<<<<<<<<<<<<<\n\n"
mkdir -p ../results/ && mkdir -p ../results/seq
$PROFILE ./PAIR 1 +$bench $IO +$bench $IO | tee ../results/seq/$bench.txt


exit 

n=0
for benchA in "${!datamap[@]}"; do 
  for benchB in "${!datamap[@]}"; do 
    if [ "$benchA" == "$benchB" ] || [[ "$benchA" > "$benchB" ]]; then
      continue
    fi
  
    # compile a new build!
    (cd .. && mkdir -p build-$benchA-$benchB && cd build-$benchA-$benchB && $benchA=1 $benchB=1 cmake .. && make)


    IO_A=$(build_input "$benchA" "${datamap[$benchA]}")
    IO_B=$(build_input "$benchB" "${datamap[$benchB]}")

    echo -e "\n\n>>>>>>>>>>>>>>>> Launching both kernels: $bench_A and $bench_B <<<<<<<<<<<<<<<<\n\n"
    $PROFILE ../build-$benchA-$benchB/PAIR b +$benchA $IO_A +$benchB $IO_B | tee results/conc/$benchA_$benchB.txt

  let "n += 1"

  done
done




