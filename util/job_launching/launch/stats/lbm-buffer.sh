#!/bin/bash

jobs=("lbm-256" "lbm-512" "lbm-1024" "lbm-2048" "lbm-4096" \
  "lbm-fixed" "lbm-hop" "lbm-clk2x")

for job in ${jobs[@]}; do

  python ../get_stats_smk.py -R ~/project/gpusim-results/lbm/run-$job \
    -N $job -S seq.yml -L ~/project/gpusim-results/logfiles \
    -o ../results/$job.csv

done
