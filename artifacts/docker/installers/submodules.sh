#!/usr/bin/env bash

set -e

mkdir -p /tmp/installers
pushd /tmp/installers

# install submodules
BASE_VERSION=54c035f7834b564019656c3e3fcc3646292f727d

git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
pushd gaussian-splatting
git reset --hard ${BASE_VERSION}

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim

popd


popd
