#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo "PLEASE SET <MAP_FOLDER> <SESSION NAME>"
    exit 1
fi

SESSION=$2
MAP_FOLDER=$1

echo "====================== PROCESS VIDEOS ======================"
export OPENCV_FFMPEG_READ_ATTEMPTS=10000
python mapmind/panorama/extract_video_images.py --input_video ${MAP_FOLDER}/${SESSION} --create_subfoler 0 --images images_pano

echo "====================== ADD GPS EXIF ======================"
python mapmind/panorama/gopro_gps_extractor.py --input_video_folder ${MAP_FOLDER}/${SESSION}/

echo "====================== RUN openMVG ======================"
bash mapmind/openMVG_pano.sh ${MAP_FOLDER}/${SESSION}/images_pano ${MAP_FOLDER}/${SESSION}/openMVG

# todo : flatten the map

echo "====================== RUN GAUSSIAN ======================"
bash mapmind/colmap/gaussian.sh ${MAP_FOLDER} ${SESSION}/openMVG/reconstruction
