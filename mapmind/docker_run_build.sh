#!/usr/bin/env bash
set -e


MODEL_DIR=$(pwd)

docker run -d --rm --name 'EasyGaussianSplatting' \
--gpus 'all,"capabilities=compute,utility,graphics,video"' \
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
-e NVIDIA_VISIBLE_DEVICES=all \
-p 8001:8001 \
-v ${MODEL_DIR}:/EasyGaussianSplatting \
colmap_gaussian_splatting \
bash -c "
cd /EasyGaussianSplatting
conda run -n gaussian_splatting ./mapmind/run_360.sh /EasyGaussianSplatting/data insta360_test
chmod -R 777 /EasyGaussianSplatting/data/insta360_test
"
