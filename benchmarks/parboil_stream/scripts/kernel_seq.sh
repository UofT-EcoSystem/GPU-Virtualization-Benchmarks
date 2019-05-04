#declare -A datamap=(["cutcp"]="large" ["histo"]="large" ["lbm"]="long" ["mri_gridding"]="small" \
  #  ["mri_q"]="large" ["sad"]="large" ["sgemm"]="medium" ["spmv"]="large" ["stencil"]="default" \
  #  ["tpacf"]="large")


kernels=("cutcp" "sgemm" "tpacf" "lbm" "sad" "spmv" "stencil") 

for bench in ${kernels[@]}; do 
  echo $bench
  tmux new-window -n:seq_$bench docker exec -it --user serinatan sim bash -c "cd /mnt/GPU-Virtualization-Benchmarks/benchmarks/parboil_stream/scripts/ && export CUDA_INSTALL_PATH=/usr/local/cuda && ./kernel_launch_stencil.sh $bench sim"
done


