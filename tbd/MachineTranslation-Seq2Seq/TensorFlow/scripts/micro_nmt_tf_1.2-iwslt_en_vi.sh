#!/bin/bash

NMT_ROOT=$(cd $(dirname $0) && pwd)/..

mkdir -p $NMT_ROOT/measurements
rm -f $NMT_ROOT/measurements/$(basename $0).nvvp

PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
  --export-profile $NMT_ROOT/measurements/$(basename $0).nvvp --print-summary"
HPARAM_FNAME="nmt_tf_1.2-iwslt15_en_vi-micro.json"

# if [ "$1" = "" ]
# then
# 	PREFIX=
# 	HPARAM_FNAME="nmt_tf_1.2-iwslt15_en_vi-micro.json"
# elif [ "$1" = "--profile" ]
# then
# 	PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
# 		--export-profile $NMT_ROOT/measurements/$(basename $0).nvvp --print-summary"
# 	HPARAM_FNAME="nmt_tf_1.2-iwslt15_en_vi-520_updates.json"
# elif [ "$1" = "--profile-fp32" ]
# then
# 	PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
# 		--export-profile $NMT_ROOT/measurements/$(basename $0).nvvp --print-summary \
# 		--metrics single_precision_fu_utilization"
# 	HPARAM_FNAME="nmt_tf_1.2-iwslt15_en_vi-520_updates.json"
# else 
# 	echo "Invalid input argument. Valid ones are --profile/--profile-fp32."; exit -1
# fi

cd $NMT_ROOT/source/micro-nmt-tf-1.2 && \
$PREFIX python2 -m nmt.nmt --src=en --tgt=vi \
                           --hparams_path=$NMT_ROOT/hparams/$HPARAM_FNAME \
                           --out_dir=$NMT_ROOT/models \
                           --vocab_prefix=$NMT_ROOT/dataset/iwslt15_en-vi/vocab \
                           --train_prefix=$NMT_ROOT/dataset/iwslt15_en-vi/train \
                           --dev_prefix=$NMT_ROOT/dataset/iwslt15_en-vi/tst2012 \
                           --test_prefix=$NMT_ROOT/dataset/iwslt15_en-vi/tst2013
