#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OCLDIR=$DIR/../cuda

# 10 in total
#  and  does not work
bm="backprop b+tree dwt2d heartwall hotspot3D kmeans hybridsort \
    leukocyte \
    nw pathfinder streamcluster bfs gaussian hotspot \
    lavaMD lud nn particlefilter srad huffman"

OUTDIR=$DIR/results-utilsust
#mkdir -f $OUTDIR &>/dev/null
mkdir -p $OUTDIR 

cd $OCLDIR
exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

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
    #nv-nsight-cu-cli -f --section-folder ../sections/ --section InstCounter --csv $cmd | tee $OUTDIR/$b.txt
    nv-nsight-cu-cli -f --section-folder ../sections/ --section PertSustainedActive --csv $cmd | tee $OUTDIR/$b.txt
    #nv-nsight-cu-cli -f --section-folder ../sections/ --section PertBurstActive --csv $cmd | tee $OUTDIR/$b.txt

    cd $OCLDIR
    #exe echo
    #echo
done
