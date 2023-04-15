#!bin/bash

apt-get update
apt-get upgrade -y
apt install nano
apt install libosmesa6-dev libgl1-mesa-glx libglfw3 -y

apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common

apt-get install -y patchelf

MUJOCOPATH=https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
DOWNLOADED_PATH=./mujoco210-linux-x86_64.tar.gz
EXTRACTED_PATH=~/.mujoco
apt-get install wget -y
wget $MUJOCOPATH
mkdir $EXTRACTED_PATH
tar -xf $DOWNLOADED_PATH -C $EXTRACTED_PATH

echo 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
source ~/.bashrc