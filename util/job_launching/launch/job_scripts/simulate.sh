#!/bin/bash

if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    export GPGPUSIM_ROOT=REPLACE_GPGPUSIM_ROOT
    source $GPGPUSIM_ROOT/setup_environment
else
    echo "Skipping setup_environment - already set"
fi

echo "doing: export benchmark root directory"
export BENCH_HOME=REPLACE_BENCH_HOME

echo "doing: export -n PTX_SIM_USE_PTX_FILE"
export -n PTX_SIM_USE_PTX_FILE
echo "doing: export LD_LIBRARY_PATH=REPLACE_LIBPATH:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=REPLACE_LIBPATH:$LD_LIBRARY_PATH

echo "doing: cd REPLACE_SUBDIR"
cd "REPLACE_SUBDIR"

echo "doing: cmake $BENCH_HOME"
if REPLACE_VALID_APP_2; then
    cmake $BENCH_HOME -DREPLACE_SHORT_APP_1=ON -DREPLACE_SHORT_APP_2=ON
else
    cmake $BENCH_HOME -DREPLACE_SHORT_APP_1=ON
fi

echo "doing: make -j4"
make -j4

echo "doing: write input files"
echo "REPLACE_INPUT_1" > REPLACE_APP_1.txt

if REPLACE_VALID_APP_2; then
    echo "REPLACE_INPUT_2" > REPLACE_APP_2.txt
fi

# Uncomment to force blocking torque launches
# echo "doing export CUDA_LAUNCH_BLOCKING=1"
# export CUDA_LAUNCH_BLOCKING=1

ldd driver | tee ldpath.txt

if REPLACE_VALID_APP_2; then
    echo "doing: ./driver REPLACE_APP_1.txt REPLACE_APP_2.txt"
    ./driver REPLACE_APP_1.txt REPLACE_APP_2.txt
else
    echo "doing: ./driver REPLACE_APP_1.txt"
    ./driver REPLACE_APP_1.txt
fi
