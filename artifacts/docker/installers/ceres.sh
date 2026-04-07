#!/usr/bin/env bash

set -e

INSTALL_PREFIX=/usr/local
if [[ ! -z $1 ]]; then
  INSTALL_PREFIX=$1
fi

mkdir -p /tmp/installers
pushd /tmp/installers

# install eigen
EIGEN_VERSION=3.4.0
wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.zip
unzip eigen-${EIGEN_VERSION}.zip

pushd eigen-${EIGEN_VERSION}
mkdir build
pushd build
cmake ..
make install
popd
popd

rm -rf eigen-${EIGEN_VERSION} eigen-${EIGEN_VERSION}.zip

echo "Install PoseLib"
git clone --recursive https://github.com/PoseLib/PoseLib.git
pushd PoseLib
mkdir build
pushd build

cmake ..
cmake --build . --target install -j$(($(nproc)-1))
popd
popd
rm -rf PoseLib


# install ceres
CERES_VERSION=2.0.0
echo "Install ceres " ${CERES_VERSION}
wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/${CERES_VERSION}.tar.gz
tar zxf ${CERES_VERSION}.tar.gz
mkdir ceres-bin
pushd ceres-bin
# export CUDA_HOME=/usr/local/cuda-11.8
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64
# export PATH=$PATH:$CUDA_HOME/bin
# export CUDACXX=/usr/local/cuda-11.8/bin/nvcc
cmake ../ceres-solver-${CERES_VERSION} -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j$(($(nproc)-1))
make install
popd

popd
