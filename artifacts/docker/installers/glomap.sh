#!/usr/bin/env bash

set -e

mkdir -p /tmp/installers
pushd /tmp/installers

echo "Install glomap"

# GLOMAP_VERSION=1.0.0
# wget https://github.com/colmap/glomap/archive/refs/tags/${GLOMAP_VERSION}.tar.gz
# tar zxf ${GLOMAP_VERSION}.tar.gz
# pushd glomap-${GLOMAP_VERSION}

git clone --recursive https://github.com/colmap/glomap
pushd glomap
git reset --hard cbb0b5ece38b1c3639d33dcc6b9374862ab5a21e

# Insert the line after line 3
# sed -i '47a '"install(TARGETS colmap)" "cmake/FindDependencies.cmake"

mkdir build
pushd build

# https://github.com/colmap/glomap/issues/55
cmake .. -GNinja -DMARCH_NATIVE=OFF -DCCACHE_ENABLED=OFF -DFETCH_COLMAP=OFF -DFETCH_POSELIB=OFF -DCCACHE_ENABLED=OFF
ninja && ninja install
popd
popd

# rm -rf ${GLOMAP_VERSION}.tar.gz
# rm -rf glomap-${GLOMAP_VERSION}
rm -rf glomap

popd
