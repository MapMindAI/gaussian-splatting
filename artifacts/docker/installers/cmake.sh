#!/usr/bin/env bash

set -e

mkdir -p /tmp/installers
pushd /tmp/installers

wget https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1.tar.gz
tar xfvz cmake-3.30.1.tar.gz
pushd cmake-3.30.1

./bootstrap -- -DCMAKE_USE_OPENSSL=OFF
make -j$(($(nproc)-1))
make install
popd

rm cmake-3.30.1.tar.gz
rm -rf cmake-3.30.1

popd
