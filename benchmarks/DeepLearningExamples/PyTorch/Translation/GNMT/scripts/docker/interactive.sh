#!/bin/bash

nvidia-docker run -it --rm --ipc=host -v $PWD:/workspace/gnmt/ -v /scratch/dataset/:/dataset/ gnmt bash
