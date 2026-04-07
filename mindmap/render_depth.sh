#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo "PLEASE SET <MAP_FOLDER> <SESSION NAME>"
    echo "example ./mindmap/render_depth.sh /mnt/data/yeliu/gaussian_splatting/GoPro test"
    exit 1
fi

MAP_FOLDER=$1
SESSION=$2

echo "Process session : " ${SESSION}

export PYTHONPATH="$PYTHONPATH:/mnt/data/yeliu/Dev/GaussianSplatting/submodules/libs"
OUTPUT_FOLDER=output

echo "====================== PROCESS GAUSSSIAN DEPTH RENDER ======================"
cd /mnt/data/yeliu/Dev/GaussianSplatting
python render_depth.py \
--source_path ${MAP_FOLDER}/${SESSION} \
--model_path  ${MAP_FOLDER}/${SESSION}/${OUTPUT_FOLDER}

echo "====================== DONE ======================"
