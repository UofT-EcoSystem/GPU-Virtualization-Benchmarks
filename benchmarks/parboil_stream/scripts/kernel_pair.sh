kernels=("cutcp" "sgemm" "tpacf" "lbm" "sad" "spmv") 

n=0
for benchA in "${kernels[@]}"; do 
  for benchB in "${kernels[@]}"; do 
    if [ "$benchA" == "$benchB" ] || [[ "$benchA" > "$benchB" ]]; then
      continue
    fi

    if [[ $n -lt 10 ]]; then
      echo $benchA, $benchB

      tmux new-window -n:$benchA$benchB docker exec -it --user serinatan sim \
        bash -c "cd /mnt/GPU-Virtualization-Benchmarks/benchmarks/parboil_stream/scripts/ && export CUDA_INSTALL_PATH=/usr/local/cuda && ./kernel_launch.sh $benchA:$benchB sim; bash -i"

    fi

    let "n += 1"

  done
done

echo $n




