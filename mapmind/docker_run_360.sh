#!/usr/bin/env bash
set -e


MODEL_DIR=$(pwd)
SESSION=$2
MAP_FOLDER=$1


docker run --rm --name 'EasyGaussianSplatting' \
--gpus 'all,"capabilities=compute,utility,graphics,video"' \
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
-e NVIDIA_VISIBLE_DEVICES=all \
-p 8001:8001 \
-v ${MODEL_DIR}:/EasyGaussianSplatting \
colmap_gaussian_splatting \
bash -c "
cd /EasyGaussianSplatting
conda run --no-capture-output -n gaussian_splatting ./mapmind/run_360.sh ${MAP_FOLDER} ${SESSION}
chmod -R 777 ${MAP_FOLDER}/${SESSION}
"
