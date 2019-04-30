#!/bin/bash
PARBOIL_ROOT="$(dirname "$(pwd)")"
PARBOIL_DATA=$PARBOIL_ROOT/datasets
NSIGHT_SECTION=$PARBOIL_ROOT/../sections

# define benchmark to dataset pair (largest available set)
#declare -A datamap=(["cutcp"]="large" ["histo"]="large" ["lbm"]="long" ["mri-gridding"]="small" \
#  ["mri-q"]="large" ["sad"]="large" ["sgemm"]="medium" ["spmv"]="large" ["stencil"]="default" \
#  ["tpacf"]="large")


declare -A datamap=(["mri-q"]="large" ["sgemm"]="medium") 

PROFILE=""


if [ "$1" == "sim" ]; then
  PARBOIL_BUILD=$PARBOIL_ROOT/build-docker
else
  PARBOIL_BUILD=$PARBOIL_ROOT/build

  if [ "$2" == "nsight" ]; then
    #PROFILE="nv-nsight-cu-cli -f --csv --section-folder $NSIGHT_SECTION --section Memory_Usage "
    PROFILE="nv-nsight-cu-cli -f --csv "
  elif [ "$2" == "nvprof" ]; then
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

n=0
for benchA in "${!datamap[@]}"; do 
  for benchB in "${!datamap[@]}"; do 
    if [ "$benchA" == "$benchB" ] || [[ "$benchA" > "$benchB" ]]; then
      continue
    fi
  
    IO_A=$(build_input "$benchA" "${datamap[$benchA]}")
    IO_B=$(build_input "$benchB" "${datamap[$benchB]}")

#    echo -e "\n\n>>>>>>>>>>>>>>>> Launching first kernel <<<<<<<<<<<<<<<<\n\n"
     gdb --args $PROFILE $PARBOIL_BUILD/PAIR b +$benchA $IO_A +$benchB $IO_B
#    echo -e "\n\n>>>>>>>>>>>>>>>> Launching second kernel <<<<<<<<<<<<<<<<\n\n"
#    echo $PROFILE $PARBOIL_BUILD/PAIR 2 +$benchA "$IO_A" +$benchB "$IO_B"
#    echo -e "\n\n>>>>>>>>>>>>>>>> Launching both kernels <<<<<<<<<<<<<<<<\n\n"
#    echo $PROFILE $PARBOIL_BUILD/PAIR b +$benchA "$IO_A" +$benchB "$IO_B"

  let "n += 1"

  done
done

echo $n



