#!/usr/bin/env bash

set -e

mkdir -p /tmp/installers
pushd /tmp/installers


# https://exiftool.org/index.html
wget https://exiftool.org/Image-ExifTool-12.55.tar.gz
gzip -dc Image-ExifTool-12.55.tar.gz | tar -xf -
cd Image-ExifTool-12.55
perl Makefile.PL
make test
make install

popd
