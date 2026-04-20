#!/usr/bin/env bash
set -e


if [ $# -lt 2 ]; then
    echo "PLEASE SET <MAP_FOLDER> <SESSION NAME>"
    exit 1
fi

MAP_FOLDER=$1
SESSION=$2

# check if the model exist
SPARSE_MODEL_PATH=${MAP_FOLDER}/${SESSION}/sparse/0/
if [ -d "$SPARSE_MODEL_PATH" ]; then
    echo "SPARSE MODEL EXIST"
    exit 0
fi

echo "====================== PROCESS FEATURE EXTRACTION ======================"

colmap feature_extractor \
--database_path ${MAP_FOLDER}/${SESSION}/database.db \
--image_path ${MAP_FOLDER}/${SESSION}/images \
--ImageReader.camera_model=PINHOLE \
--ImageReader.single_camera_per_folder 1

echo "====================== PROCESS SPATIAL MATCHER ======================"


colmap spatial_matcher \
--database_path ${MAP_FOLDER}/${SESSION}/database.db \
--TwoViewGeometry.min_inlier_ratio 0.2 \
--SpatialMatching.max_num_neighbors 200 \
--SpatialMatching.max_distance 100 \
--SpatialMatching.ignore_z 0

echo "====================== PROCESS GLOMAP MAPPER ======================"

glomap mapper \
--database_path ${MAP_FOLDER}/${SESSION}/database.db \
--output_path ${MAP_FOLDER}/${SESSION}/sparse_raw \
--image_path ${MAP_FOLDER}/${SESSION}/images \
--RelPoseEstimation.max_epipolar_error 8

echo "====================== PROCESS IMAGE UNDISTORTER ======================"

# mkdir -p {target_directory}/dense/sparse/0
# colmap model_converter \
# --input_path ${MAP_FOLDER}/${SESSION}/sparse/0 \
# --output_path ${MAP_FOLDER}/${SESSION}/sparse/0 \
# --output_type TXT

python mindmap/colmap/transform_colmap_model.py \
--database_path ${MAP_FOLDER}/${SESSION}/database.db \
--model_path ${MAP_FOLDER}/${SESSION}/sparse_raw/0 \
--output_model_path ${MAP_FOLDER}/${SESSION}/sparse/0
