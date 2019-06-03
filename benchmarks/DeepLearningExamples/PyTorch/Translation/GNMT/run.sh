# Usage
# run.sh [train|infer] [util|inst] GPU_ID

export CUDA_VISIBLE_DEVICES=$3

nsight_path=/usr/local/NVIDIA-Nsight-Compute-2019.3/

train="./train.py --seed 2 --train-batch-size 16 --epochs 1 --profile 10"
infer="/opt/conda/bin/python3 translate.py \
  --input data/wmt16_de_en/newstest2014.tok.bpe.32000.en \
  --reference data/wmt16_de_en/newstest2014.de --output /tmp/output \
  --model results/gnmt/model_best.pth --batch-size 32 \
  --beam-size 5 --math fp16"

util_sec="PertSustainedActive"
inst_sec="InstCounter"

if [ $1 == "train" ]; then
  cmd=$train
else
  cmd=$infer
fi

if [ $2 == "util" ]; then
  section=$util_sec
else
  section=$inst_sec
fi

# training: comp unit utilization
$nsight_path/nv-nsight-cu-cli --target-processes all --section-folder sections/ --section $section \
  --csv -f --profile-from-start off $cmd | tee profile/$1-$2.txt

