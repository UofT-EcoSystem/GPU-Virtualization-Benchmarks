PARBOIL_ROOT="$(dirname "$(pwd)")"
PARBOIL_DATA=$PARBOIL_ROOT/datasets
NSIGHT_SECTION=$PARBOIL_ROOT/../sections

SGEMM_MED=$PARBOIL_DATA/sgemm/medium/input
SGEMM_SMALL=$PARBOIL_DATA/sgemm/small/input

SPMV_LARGE=$PARBOIL_DATA/spmv/large/input
SPMV_MED=$PARBOIL_DATA/spmv/medium/input
SPMV_SMALL=$PARBOIL_DATA/spmv/small/input


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


#set up sgemm inputs
folder=$SGEMM_MED
desc=$(cat $folder/DESCRIPTION)
empty=true
SGEMM_IN=""

for i in $(echo $desc | tr " " "\n"); do 
  if [ "$i" == "Inputs:" ]; then
    continue
  fi

  if [ "$empty" = false ]; then
    SGEMM_IN="$SGEMM_IN,"
  fi

  SGEMM_IN="$SGEMM_IN$folder/$i"

  empty=false

done

SGEMM_IN="$SGEMM_IN -o $PARBOIL_BUILD/out/sgemm/matrix.out"

#set up spmv inputs
folder=$SPMV_LARGE
desc=$(cat $folder/DESCRIPTION)
empty=true
SPMV_IN=""

for i in $(echo $desc | tr " " "\n"); do 
  if [ "$i" == "Inputs:" ]; then
    continue
  fi

  if [ "$empty" = false ]; then
    SPMV_IN="$SPMV_IN,"
  fi

  SPMV_IN="$SPMV_IN$folder/$i"

  empty=false

done

SPMV_IN="$SPMV_IN -o $PARBOIL_BUILD/out/spmv/matrix.out"

echo $SPMV_IN
echo $SGEMM_IN


#echo ">>>>>>>>>>>>>>>> Launching first kernel"
#$PROFILE $PARBOIL_BUILD/PAIR 1 --spmv -i $SPMV_IN --sgemm -i $SGEMM_IN
#echo ">>>>>>>>>>>>>>>> Launching second kernel"
#$PROFILE $PARBOIL_BUILD/PAIR 2 --spmv -i $SPMV_IN --sgemm -i $SGEMM_IN
$PROFILE $PARBOIL_BUILD/PAIR b +spmv -i $SPMV_IN +sgemm -i $SGEMM_IN

#$PROFILE $PARBOIL_BUILD/PAIR --spmv -i $PARBOIL_DATA/spmv/medium/input/bcsstk18.mtx,$PARBOIL_DATA/spmv/medium/input/vector.bin --sgemm -i $PARBOIL_DATA/sgemm/small/input/matrix1.txt,$PARBOIL_DATA/sgemm/small/input/matrix2t.txt,$PARBOIL_DATA/sgemm/small/input/matrix2t.txt 

#$PARBOIL_ROOT/build-docker/PAIR --sgemm -i $PARBOIL_DATA/sgemm/small/input/matrix1.txt,$PARBOIL_DATA/sgemm/small/input/matrix2t.txt,$PARBOIL_DATA/sgemm/small/input/matrix2t.txt -o $PARBOIL_ROOT/benchmarks/sgemm/run/small/matrix3.txt --spmv -i $PARBOIL_DATA/spmv/medium/input/bcsstk18.mtx,$PARBOIL_DATA/spmv/medium/input/vector.bin -o $PARBOIL_ROOT/benchmarks/spmv/run/medium/Dubcova3.mtx.out

#$PARBOIL_ROOT/build-docker/PAIR --sgemm -i $PARBOIL_DATA/sgemm/medium/input/matrix1.txt,$PARBOIL_DATA/sgemm/medium/input/matrix2t.txt,$PARBOIL_DATA/sgemm/medium/input/matrix2t.txt -o $PARBOIL_ROOT/benchmarks/sgemm/run/medium/matrix3.txt --spmv -i $PARBOIL_DATA/spmv/small/input/1138_bus.mtx,$PARBOIL_DATA/spmv/small/input/vector.bin -o $PARBOIL_ROOT/benchmarks/spmv/run/small/Dubcova3.mtx.out

#-o $PARBOIL_ROOT/benchmarks/spmv/run/medium/Dubcova3.mtx.out 
#-o $PARBOIL_ROOT/benchmarks/sgemm/run/small/matrix3.txt 
