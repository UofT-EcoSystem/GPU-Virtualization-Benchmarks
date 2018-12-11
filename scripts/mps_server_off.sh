#!/bin/bash

echo "Stopping MPS server..."
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 0 -c DEFAULT
