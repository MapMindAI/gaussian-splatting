#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo "PLEASE SET <MAP_FOLDER> <SESSION NAME>"
    exit 1
fi

SESSION=$2
MAP_FOLDER=$1

echo "====================== PROCESS VIDEOS ======================"
python dm/panorama/extract_pinhole_images.py --input_video ${MAP_FOLDER}/${SESSION} --process_interval 10 --num_divide 6

export OPENCV_FFMPEG_READ_ATTEMPTS=10000
python dm/panorama/extract_video_images.py --input_video ${MAP_FOLDER}/${SESSION} --ignore_header GS

echo "====================== RUN COLMAP ======================"
# if qrcodes are needed to recover scale and transform, add 3rd param 1. Make sure you have correct qr.json
# bash dm/colmap/run_video.sh ${MAP_FOLDER} ${SESSION} 1
bash dm/colmap/run_video.sh ${MAP_FOLDER} ${SESSION}

echo "====================== RUN GAUSSIAN ======================"
bash dm/colmap/gaussian.sh ${MAP_FOLDER} ${SESSION}

echo "====================== RUN GAUSSIAN DEPTH ======================"
bash dm/render_depth.sh ${MAP_FOLDER} ${SESSION}

echo "====================== RUN TSDF MESH ======================"
python dm/tsdf/tsdf_modeling.py --model_path ${MAP_FOLDER}/${SESSION}

echo "====================== RUN DATABASE PREPARE ======================"
python dm/vlp/update_map_database.py --map_dir ${MAP_FOLDER}/${SESSION}
python dm/vlp/make_retrieval_db.py --model_path ${MAP_FOLDER}/${SESSION}