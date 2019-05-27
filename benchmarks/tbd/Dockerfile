FROM tensorflow/tensorflow:1.13.1-gpu-py3
RUN apt-get update && \
apt-get install -y --no-install-recommends \
vim-gnome g++ make pkg-config \
cuda-nsight-compute-10-1 \
virtualenv \
libopencv-dev \
libopenblas-dev \
libjemalloc-dev \
python3-dev \
python3-numpy \
python3-pip \
python3-six \
python3-setuptools && \
rm -rf /var/lib/apt/lists/* # installation
