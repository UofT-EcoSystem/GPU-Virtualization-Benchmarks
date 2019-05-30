FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
RUN apt-get update && \
apt-get install -y --no-install-recommends \
virtualenv \
sudo \
vim g++ make pkg-config \
libopencv-dev \
libopenblas-dev \
libjemalloc-dev \
python3-dev \
python3-pip \
python3-setuptools && \
rm -rf /var/lib/apt/lists/* # installation
# Update UID and GID as needed
RUN adduser --home /home/serinatan --shell /bin/bash --uid 3724 --gid 3724 --gecos '' \
--disabled-password serinatan 
RUN adduser serinatan sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER serinatan
WORKDIR /home/serinatan


