#!/usr/bin/env bash

set -e

mkdir -p /tmp/installers
pushd /tmp/installers


# get the shared libraries
# https://drive.google.com/file/d/1-zadqgthviMRZD3lAyPJ_4r4Of7rBH-f/view?usp=share_link
# download the file from google drive
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget \
--quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1-zadqgthviMRZD3lAyPJ_4r4Of7rBH-f' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-zadqgthviMRZD3lAyPJ_4r4Of7rBH-f" -O MediaSDK.zip && rm -rf cookies.txt

unzip MediaSDK.zip

# copy all the shared libraries to /usr/local/lib
cp MediaSDK/lib/*.so /usr/local/lib/
cp MediaSDK/lib/libtbb.so /usr/lib/x86_64-linux-gnu/libtbb.so.2
cp -r MediaSDK/include /usr/local/include/MediaSDK

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
g++ MediaSDK/example/main.cc -std=c++11 -I/usr/local/include/MediaSDK \
-L/usr/local/lib -lMediaSDK -lpthread -ltbb -lcuda -lMNN  -o /usr/local/bin/insta360_media_stitcher


popd
