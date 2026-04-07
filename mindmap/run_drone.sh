#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo "PLEASE SET <MAP_FOLDER> <SESSION NAME>"
    exit 1
fi

MAP_FOLDER=$1
SESSION=$2
echo "WORK DIR is :"
echo ${MAP_FOLDER}/${SESSION}

echo "====================== PROCESS VIDEOS ======================"
python dm/colmap/drone_image_extractor.py --input_video ${MAP_FOLDER}/${SESSION} --image_interval_sec 2

echo "====================== RUN COLMAP ======================"
bash dm/colmap/run_drone.sh ${MAP_FOLDER} ${SESSION}

echo "====================== RUN GAUSSIAN ======================"
bash dm/colmap/gaussian.sh ${MAP_FOLDER} ${SESSION} $3
