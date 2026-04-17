#!/usr/bin/env bash

set -e

mkdir -p /tmp/installers
pushd /tmp/installers

# https://exiftool.org/index.html
wget -O Image-ExifTool-13.56.tar.gz https://sourceforge.net/projects/exiftool/files/Image-ExifTool-13.56.tar.gz/download 
gzip -dc Image-ExifTool-13.56.tar.gz | tar -xf -
cd Image-ExifTool-13.56
perl Makefile.PL
make test
make install

popd
