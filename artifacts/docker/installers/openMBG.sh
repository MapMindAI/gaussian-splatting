#!/usr/bin/env bash

set -e

mkdir -p /tmp/installers
pushd /tmp/installers

echo "Install openMVG 1fc65e93508bd19bb7a3d7ae0bf2c8c2029404bc"

git clone --recursive https://github.com/openMVG/openMVG.git
pushd openMVG
git reset --hard 1fc65e93508bd19bb7a3d7ae0bf2c8c2029404bc
popd

mkdir openMVG_Build
pushd openMVG_Build

cmake -DCMAKE_BUILD_TYPE=RELEASE \
  -DOpenMVG_BUILD_EXAMPLES=OFF \
  -DOpenMVG_BUILD_DOC=OFF \
  -DOpenMVG_BUILD_GUI_SOFTWARES=OFF \
  -DOpenMVG_USE_RERUN=OFF \
  ../openMVG/src
make -j$(($(nproc)-1))
make install

popd

rm -rf openMVG
rm -rf openMVG_Build

popd
