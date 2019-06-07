# Usage
# run.sh [train|infer] [util|inst] GPU_ID

export CUDA_VISIBLE_DEVICES=$3

nsight_path=/usr/local/NVIDIA-Nsight-Compute-2019.3/

train="./train.py --seed 2 --train-batch-size 64 --epochs 1 --profile 10 --dataset-dir /dataset/wmt_ende"
infer="/opt/conda/bin/python3 translate.py \
  --input /dataset/wmt_ende/newstest2014.tok.bpe.32000.en \
  --reference /dataset/wmt_ende/newstest2014.de --output /tmp/output \
  --model results/gnmt/model_best.pth --batch-size 32 \
  --beam-size 5 --math fp16"

util_sec="PertSustainedActive"
inst_sec="InstCounter"
mem_sec="Memory_Usage"

if [ $1 == "train" ]; then
  cmd=$train
else
  cmd=$infer
fi

if [ $2 == "util" ]; then
  section=$util_sec
elif [ $2 == "inst" ]; then
  section=$inst_sec
else
  section=$mem_sec
fi

mkdir -p profile
mkdir -p profile/$2

# training: comp unit utilization
$nsight_path/nv-nsight-cu-cli --target-processes all --section-folder ../../../../sections/ --section $section \
  --csv -f --profile-from-start off $cmd | tee profile/$2/$1.txt

