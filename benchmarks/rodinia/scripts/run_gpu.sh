#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "run_gpu.sh [outdir]"
    exit
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OCLDIR=$DIR/../cuda
nsight="/usr/local/cuda/NsightCompute-2019.3/nv-nsight-cu-cli"
nvprof="/usr/local/cuda/bin/nvprof"

# 10 in total
#  and  does not work
bm="backprop b+tree dwt2d heartwall hotspot3D kmeans hybridsort \
    leukocyte \
    nw pathfinder streamcluster bfs gaussian hotspot \
    lavaMD lud nn particlefilter srad huffman"
#bm="particlefilter"

#OUTDIR=$DIR/results-utilsust
#mkdir -f $OUTDIR &>/dev/null
OUTDIR=$DIR/$1
mkdir -p $OUTDIR/comp
mkdir -p $OUTDIR/inst
mkdir -p $OUTDIR/mem
mkdir -p $OUTDIR/time

cd $OCLDIR
#exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    #$@ |& tee -a $OUTDIR/$b.txt ; }

for b in $bm; do
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $b
    #make clean && make TYPE=GPU

    #for idx in `seq 1 15`; do
    #    exe ./run -p 1 -d 0
    #    exe echo
    #done
    cmd=$(cat run)
    #$nsight -f --section-folder $DIR/../../sections/ --section PertSustainedActive --csv $cmd | tee $OUTDIR/comp/$b.txt
    #$nsight -f --section-folder $DIR/../../sections/ --section Memory_Usage --csv $cmd | tee $OUTDIR/mem/$b.txt
    #$nsight -f --section-folder $DIR/../../sections/ --section InstCounter --csv $cmd | tee $OUTDIR/inst/$b.txt

    $nvprof -f --print-gpu-trace --csv --normalized-time-unit ms --log-file $OUTDIR/time/$b.txt $cmd
    cd $OCLDIR
    #exe echo
    #echo
done
