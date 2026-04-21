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

export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

echo "====================== PROCESS VIDEOS ======================"
python mapmind/colmap/drone_image_extractor.py --input_video ${MAP_FOLDER}/${SESSION} --image_interval_sec 1

echo "====================== RUN COLMAP ======================"
bash mapmind/colmap/run_drone.sh ${MAP_FOLDER} ${SESSION}

echo "====================== RUN GAUSSIAN ======================"
bash mapmind/colmap/gaussian.sh ${MAP_FOLDER} ${SESSION} $3
