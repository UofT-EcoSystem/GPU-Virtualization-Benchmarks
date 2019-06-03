#!/bin/bash

python3 -m torch.distributed.launch --nproc_per_node=8 qa/qa_accuracy_main.py --bs 64 --fp16 --warmup 300 --learning-rate 2.6e-3 --seed 1 --benchmark-mode epoch-accuracy --benchmark-file qa/curve_baselines/SSD300_pytorch_19.01_fp16_1epoch_run_acc_baseline.json --data $1

