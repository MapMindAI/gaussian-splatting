#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo "PLEASE SET <MAP_FOLDER> <SESSION NAME>"
    exit 1
fi

if [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == CYGWIN* ]] || [[ "$(uname -s)" == MSYS* ]]; then
    MODEL_DIR=$(pwd -W)
else
    MODEL_DIR=$(pwd)
fi
SESSION=$2
MAP_FOLDER=$1


docker run --rm --name 'EasyGaussianSplatting' \
--gpus 'all,"capabilities=compute,utility,graphics,video"' \
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
-e NVIDIA_VISIBLE_DEVICES=all \
-p 8001:8001 \
-v ${MODEL_DIR}:/EasyGaussianSplatting \
ghcr.io/mapmindai/gaussiansplatting:sha-12e9065 \
bash -c "
cd /EasyGaussianSplatting
conda run --no-capture-output -n gaussian_splatting ./mapmind/run_drone.sh ${MAP_FOLDER} ${SESSION}
chmod -R 777 ${MAP_FOLDER}/${SESSION}
"
