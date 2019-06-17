#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "run.sh [1|2]"
    exit
fi

CURRENT_FOLD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT=$CURRENT_FOLD/..

nsight=/usr/local/cuda/NsightCompute-2019.3/nv-nsight-cu-cli
nvprof=/usr/local/cuda/bin/nvprof

sec_folder=$ROOT/../sections
comp_sec="PertSustainedActive"
inst_sec="InstCounter"
mem_sec="Memory_Usage"


if [ $1 == '1' ]; then
  # run stage 1
  ./gerbil -x 1 -x b -g -i $ROOT/dataset/Homo_sapiens.GRCh37.dna.alt.fa.gz $ROOT/tmp/

else
  mkdir -p $ROOT/profile

  # copy stage 1 data
  cp ../stage1/* ../tmp/
  $nsight --csv -f --section-folder $sec_folder --section $comp_sec -k "addKernel" -c 5 -s 2 \
    $ROOT/build/gerbil -x 2 -g -i $ROOT/tmp/ outfile | tee $ROOT/profile/comp.txt

  # copy stage 1 data
  cp ../stage1/* ../tmp/
  $nsight --csv -f --section-folder $sec_folder --section $inst_sec -k "addKernel" -c 5 -s 2 \
    $ROOT/build/gerbil -x 2 -g -i $ROOT/tmp/ outfile > $ROOT/profile/inst.txt

  # copy stage 1 data
  cp ../stage1/* ../tmp/
  $nsight --csv -f --section-folder $sec_folder --section $mem_sec -k "addKernel" -c 5 -s 2 \
    $ROOT/build/gerbil -x 2 -g -i $ROOT/tmp/ outfile > $ROOT/profile/mem.txt

  # copy stage 1 data
  cp ../stage1/* ../tmp/
  $nsight --csv -f --section-folder $sec_folder --section $comp_sec -k "agent" -c 5 -s 2 \
    $ROOT/build/gerbil -x 2 -g -i $ROOT/tmp/ outfile | tee $ROOT/profile/comp2.txt

  # copy stage 1 data
  cp ../stage1/* ../tmp/
  $nsight --csv -f --section-folder $sec_folder --section $inst_sec -k "agent" -c 5 -s 2 \
    $ROOT/build/gerbil -x 2 -g -i $ROOT/tmp/ outfile > $ROOT/profile/inst2.txt

  # copy stage 1 data
  cp ../stage1/* ../tmp/
  $nsight --csv -f --section-folder $sec_folder --section $mem_sec -k "agent" -c 5 -s 2 \
    $ROOT/build/gerbil -x 2 -g -i $ROOT/tmp/ outfile > $ROOT/profile/mem2.txt



  cp ../stage1/* ../tmp/
  $nvprof --csv -f --print-gpu-trace --normalized-time-unit ms \
    --log-file $ROOT/profile/time.txt $ROOT/build/gerbil -x 2 -g -i $ROOT/tmp/ outfile 

fi


