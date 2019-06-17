#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "run.sh [1|2|3]"
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


if [ $1 == "1" ]; then
  # run stage 1
  # CPU only
  cd $ROOT/R/S_cerevisiae

  $ROOT/bin/AriocE AriocE.gapped.cfg
  $ROOT/bin/AriocE AriocE.nongapped.cfg
elif [ $1 == "2" ]; then
  cd $ROOT/Q/S_cerevisiae

  $ROOT/bin/AriocE AriocE.paired.cfg
else
  mkdir -p $ROOT/profile

  cd $ROOT/A/S_cerevisiae
  pwd

  $nvprof --csv -f --print-gpu-trace --normalized-time-unit ms \
    --log-file $ROOT/profile/time.txt $ROOT/bin/AriocP AriocP.paired.cfg 

  $nsight --csv -f --section-folder $sec_folder --section $comp_sec \
    $ROOT/bin/AriocP AriocP.paired.cfg | tee $ROOT/profile/comp.txt

  $nsight --csv -f --section-folder $sec_folder --section $inst_sec \
    $ROOT/bin/AriocP AriocP.paired.cfg | tee $ROOT/profile/inst.txt

  $nsight --csv -f --section-folder $sec_folder --section $mem_sec \
    $ROOT/bin/AriocP AriocP.paired.cfg | tee $ROOT/profile/mem.txt


fi


