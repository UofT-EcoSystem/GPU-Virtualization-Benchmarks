#!/bin/bash

python ./qa/benchmark_performance.py --benchmark-mode training --ngpus 1 4 --bs 2 4 8 16 32 --fp16  --bench-warmup 100 --bench-iterations 200 --benchmark-file qa/benchmark_baselines/SSD300_pytorch_19.01_training_fp16.json --data $1
