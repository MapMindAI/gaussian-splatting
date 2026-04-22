#!/usr/bin/env bash
set -e


MODEL_DIR=$(pwd)

docker run -d --gpus all --rm --name 'EasyGaussianSplatting' \
-p 8001:8001 \
-v ${MODEL_DIR}:/EasyGaussianSplatting \
ghcr.io/mapmindai/gaussiansplatting:latest \
bash -c "
conda activate gaussian_splatting
./mapmind/run_360.sh /EasyGaussianSplatting/data insta360_test

chmod -R 777 /EasyGaussianSplatting/data/insta360_test
"


# docker exec -it tritonserver bash
