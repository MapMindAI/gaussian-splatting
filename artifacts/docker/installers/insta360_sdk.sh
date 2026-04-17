#!/usr/bin/env bash

set -e

mkdir -p /tmp/installers
pushd /tmp/installers

SDK_NAME="libMediaSDK-dev.deb"

apt-get update

if [ -f "${SDK_NAME}" ]; then
  echo "File exists"
else
  apt-get install -y curl
  curl -L -o ${SDK_NAME} \
    https://github.com/MapMindAI/EasyGaussianSplatting/releases/download/v0/Insta360SDK.deb
fi

apt-get install -y ./${SDK_NAME}

# rm -rf /var/lib/apt/lists/*
g++ insta_360_main.cc -std=c++11 -lMediaSDK -lpthread -lcuda -lMNN -o /usr/local/bin/insta360_media_stitcher
echo "Build done!"

apt-get remove libMediaSDK-dev

rm -rf /var/lib/apt/lists/*

popd
