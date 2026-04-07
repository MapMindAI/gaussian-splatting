#!/usr/bin/env bash
set -e


if [ $# -lt 2 ]; then
    echo "PLEASE SET <MAP_FOLDER> <SESSION NAME>"
    exit 1
fi

SESSION=$2
MAP_FOLDER=$1
USE_QRCODE_PRIOR=${3:-0}

# check if the model exist
SPARSE_MODEL_PATH=${MAP_FOLDER}/${SESSION}/sparse/0/
if [ -d "$SPARSE_MODEL_PATH" ]; then
    echo "SPARSE MODEL EXIST"
    exit 0
fi

# check if ordinary video exist, which require OPENCV camera model
TEST_PATH=${MAP_FOLDER}/${SESSION}/images/use_opencv_model/
TEST_PATH_GPS=${MAP_FOLDER}/${SESSION}/images/image_with_gps/
CAMERA_MODEL=PINHOLE
if [ -d "$TEST_PATH" ]; then
    echo "USE OPENCV MODEL"
    CAMERA_MODEL=OPENCV
fi


MASK_DIR=${MAP_FOLDER}/${SESSION}/masks
GENERATE_MASKS=1  # 设为1表示自动生成 masks, 0 表示跳过（你已生成）
# 如果开启自动生成，调用 Python 脚本
if [ "${GENERATE_MASKS}" -eq 1 ]; then
    echo "=== Generating QR masks into ${MASK_DIR} ==="
    python dm/colmap/qrcode/generate_qr_mask.py \
        --image_dir ${MAP_FOLDER}/${SESSION}/images \
        --mask_dir ${MASK_DIR} \
        --detect_model_path dm/colmap/qrcode/model/best-yolov8.pt
fi

echo "====================== PROCESS FEATURE EXTRACTION " ${CAMERA_MODEL}" ======================"

TEST_PATH_VIDEO=${MAP_FOLDER}/${SESSION}/images/no_ordinary_video/
SHARE_CAMERA="--ImageReader.single_camera_per_folder 1"
if [ -d "$TEST_PATH_VIDEO" ]; then
    echo "USE SINGLE CAMERA"
    SHARE_CAMERA="--ImageReader.single_camera 1"
fi

colmap feature_extractor ${SHARE_CAMERA} \
--database_path ${MAP_FOLDER}/${SESSION}/database.db \
--image_path ${MAP_FOLDER}/${SESSION}/images \
--ImageReader.camera_model=${CAMERA_MODEL} \
--ImageReader.mask_path ${MASK_DIR}

echo "====================== PROCESS VIDEO MATCHER ======================"

MOUNT_FOLDER=/mnt/data
if [ ! -d "$MOUNT_FOLDER" ]; then
    MOUNT_FOLDER=/mnt/ml-experiment-data
fi

NUM_IMAGES=$(find ${MAP_FOLDER}/${SESSION}/images -type f | wc -l)
if [ -d "$TEST_PATH_GPS" ]; then
    echo "====================== spatial_matcher ======================"

    colmap spatial_matcher \
    --database_path ${MAP_FOLDER}/${SESSION}/database.db \
    --TwoViewGeometry.min_inlier_ratio 0.2 \
    --SpatialMatching.max_num_neighbors 50 \
    --SpatialMatching.max_distance 40 \
    --SpatialMatching.ignore_z 1
fi
if ((${NUM_IMAGES} < 500)); then
    echo "====================== exhaustive_matcher ======================"
    colmap exhaustive_matcher \
    --database_path ${MAP_FOLDER}/${SESSION}/database.db
else
    COLMAP_VOC_PATH_32=${MOUNT_FOLDER}/yeliu/models/vocab_tree_flickr100K_words32K.bin
    COLMAP_VOC_PATH_256=${MOUNT_FOLDER}/yeliu/models/vocab_tree_flickr100K_words256K.bin
    VOC_PATH=${COLMAP_VOC_PATH_32}
    if ((${NUM_IMAGES} > 1500)); then
        VOC_PATH=${COLMAP_VOC_PATH_256}
    fi
    echo ${VOC_PATH}
    echo "====================== sequential_matcher ======================"
    colmap sequential_matcher \
    --database_path ${MAP_FOLDER}/${SESSION}/database.db \
    --TwoViewGeometry.min_inlier_ratio 0.2 \
    --SequentialMatching.vocab_tree_path ${VOC_PATH} \
    --SequentialMatching.loop_detection 1 \
    --SequentialMatching.loop_detection_num_images 50 \
    --SequentialMatching.loop_detection_num_nearest_neighbors 20
fi

echo "====================== PROCESS GLOMAP MAPPER ======================"

glomap mapper \
--database_path ${MAP_FOLDER}/${SESSION}/database.db \
--output_path ${MAP_FOLDER}/${SESSION}/sparse_raw \
--image_path ${MAP_FOLDER}/${SESSION}/images \
--ViewGraphCalib.thres_two_view_error 2 \
--RelPoseEstimation.max_epipolar_error 2 \
--BundleAdjustment.max_num_iterations 200

# rotate the model to fit UE axis
echo "====================== ROTATE COLMAP TO UE ======================"
python dm/colmap/rotate_colmap_to_UE.py \
--colmap_model_path ${MAP_FOLDER}/${SESSION} \
--input_sparse_path sparse_raw/0/ \
--output_sparse_path sparse_raw/0/

mkdir -p ${MAP_FOLDER}/${SESSION}/sparse/0
if [ -f "${MAP_FOLDER}/${SESSION}/qr_poses.json"]; then
  echo "Find prior qr_poses.json, set USE_QRCODE_PRIOR = 1"
  USE_QRCODE_PRIOR=1
fi

if [ "${USE_QRCODE_PRIOR}" == "1" ]; then
  echo "====================== USE QR-CODE PRIOR TRANSFORM ======================"
  python dm/colmap/qrcode/transform_colmap_model_qrcode.py \
  --colmap_model_dir ${MAP_FOLDER}/${SESSION} \
  --image_dir images \
  --qr_json_file dm/colmap/qrcode/qr.json \
  --prior_qr_poses_json_file qr_poses_prior.json \
  --output_dir ${MAP_FOLDER}/${SESSION}/output \
  --input_sparse_path sparse_raw/0/ \
  --output_sparse_path sparse/0/
elif [ -d "$TEST_PATH_GPS" ]; then
  echo "====================== USE GPS TRANSFORM ======================"
  python dm/colmap/transform_colmap_model.py \
  --database_path ${MAP_FOLDER}/${SESSION}/database.db \
  --model_path ${MAP_FOLDER}/${SESSION}/sparse_raw/0 \
  --output_model_path ${MAP_FOLDER}/${SESSION}/sparse/0
else
  echo "====================== USE DEFAULT MODEL CONVERTER ======================"
  colmap model_converter \
  --input_path ${MAP_FOLDER}/${SESSION}/sparse_raw/0 \
  --output_path ${MAP_FOLDER}/${SESSION}/sparse/0 \
  --output_type TXT
fi

if [ -d "$TEST_PATH" ]; then
  echo "====================== PROCESS IMAGE UNDISTORTER ======================"
  colmap image_undistorter \
  --image_path ${MAP_FOLDER}/${SESSION}/images \
  --input_path ${MAP_FOLDER}/${SESSION}/sparse/0 \
  --output_path ${MAP_FOLDER}/${SESSION}/dense \
  --output_type COLMAP \
  --max_image_size 4000

  mkdir -p ${MAP_FOLDER}/${SESSION}/dense/sparse/0
  colmap model_converter \
  --input_path ${MAP_FOLDER}/${SESSION}/dense/sparse \
  --output_path ${MAP_FOLDER}/${SESSION}/dense/sparse/0 \
  --output_type TXT
fi
