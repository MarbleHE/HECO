#!/bin/bash
# ████████████████████████████████████████████████████████████████████
# █████████████████ EVALUATION PLATFORM SETUP SCRIPT █████████████████
# ████████████████████████████████████████████████████████████████████

HOME=/home/ubuntu/
AST_OPTIMIZER_EVAL_BRANCH=seal-bfv-integration

cd $HOME
sudo apt-get update

# install some utilities
sudo apt-get -y install git
sudo apt-get -y install wget
sudo apt-get -y install make

# SEAL installation derived from
# https://github.com/MarbleHE/SHEEP/blob/master/backend/Dockerfile_sheep_base

### get gcc-9 (gcc >=6 needed for SEAL). # 7 produces errors.
# apt-get -y install software-properties-common
# add-apt-repository -y  ppa:ubuntu-toolchain-r/test
# apt-get update; apt-get -y install gcc-9 g++-9
sudo apt-get update; sudo apt-get -y install build-essential
# update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 10
# update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 10
# update-alternatives --set gcc /usr/bin/gcc-9
# update-alternatives --set g++ /usr/bin/g++-9

## get clang++ as this is faster than gcc for SEAL
sudo apt-get -y install clang-tools-9

###### build cmake from source (to get a new enough version for SEAL)
wget https://cmake.org/files/v3.15/cmake-3.15.0.tar.gz
tar -xvzf cmake-3.15.0.tar.gz
cd cmake-3.15.0; export CC=gcc-9; export CXX=g++-9; ./bootstrap; make -j8; make install

###### build optional SEAL dependencies (improve performance)
sudo apt-get -y install libmsgsl-dev
sudo apt-get -y install zlib1g-dev
###### build SEAL
git clone https://github.com/microsoft/SEAL.git
cd SEAL && export CC=clang-9 && export CXX=clang++-9 && cmake . -DSEAL_BUILD_SEAL_C=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && make -j64 && make install

# clone AST optimizer repository
cd $HOME && git clone https://github.com/MarbleHE/AST-Optimizer.git && cd AST-Optimizer
git checkout $AST_OPTIMIZER_EVAL_BRANCH
mkdir build && cd build
cmake .. && make -j64

# create directories for benchmarking
mkdir ${HOME}/benchmark_results
mkdir ${HOME}/benchmark_configs
