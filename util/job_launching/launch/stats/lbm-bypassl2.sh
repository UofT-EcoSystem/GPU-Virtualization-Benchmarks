#!/bin/bash

jobs=("lbm-4096" "lbm-bypassl2d" "lbm-bypassl2d-nortw" "lbm-nortw" "lbm-bypassl2d-seprw" "lbm-seprw")

for job in ${jobs[@]}; do

  python get_stats_smk.py -R ~/project/gpusim-results/lbm/run-$job \
    -N $job -S seq.yml -L ~/project/gpusim-results/logfiles \
    -o ../results/$job.csv

done
