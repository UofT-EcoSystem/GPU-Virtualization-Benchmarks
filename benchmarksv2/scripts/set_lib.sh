#!/bin/bash

# usage: set_lib.sh new_lib_dirpath rel/dbg

LIBDIR_ROOT=/mnt/ecosystem-gpgpu-sim/lib/gcc-5.4.0/cuda-9000
SO_NAME=libcudart.so.9.0

# make dir for lib
mkdir -p $1

if [ $? -ne 0 ]; then
  echo "make dir for new lib path failed."
  exit 1
fi

# copy lib to new lib dirpath
if [ $2 = 'rel' ]; then
  LIBDIR=$LIBDIR_ROOT/release
else
  LIBDIR=$LIBDIR_ROOT/debug
fi

cp $LIBDIR/$SO_NAME $1

if [ $? -ne 0 ]; then
  echo "copy lib failed."
  exit 1
fi

# get abs path for the new lib path
abs_path=`realpath $1`
echo $abs_path

# modify ld lib path
export LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | sed -re 's#'$GPGPUSIM_ROOT'\/lib\/[0-9]+\/(debug|release):##'`
export LD_LIBRARY_PATH=$abs_path:$LD_LIBRARY_PATH
