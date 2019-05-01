#declare -A datamap=(["cutcp"]="large" ["histo"]="large" ["lbm"]="long" ["mri_gridding"]="small" \
  #  ["mri_q"]="large" ["sad"]="large" ["sgemm"]="medium" ["spmv"]="large" ["stencil"]="default" \
  #  ["tpacf"]="large")


kernels=("cutcp" "sgemm" "tpacf" "lbm" "sad" "spmv") 

for bench in ${kernels[@]}; do 
  echo $bench
  tmux new-window -n:seq_$bench ./kernel_launch.sh $bench hw; bash -i
done


