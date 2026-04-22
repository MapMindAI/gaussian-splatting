#!/usr/bin/env bash
set -e

USE_DEPTH=false
USE_SEGMENT=false
USE_STYLE=false
STYLE_NAME=

if [ $# -lt 1 ]; then
    echo "PLEASE SET SESSION NAME"
    exit 1
fi

MAP_FOLDER=$1
SESSION=$2

if [ $# -ge 3 ]; then
    if [ "$3" = depth ] ; then
        USE_DEPTH=true
    fi
    if [ "$3" = segment ] ; then
        USE_SEGMENT=true
    fi
    if [ "$3" = depth_and_segment ] ; then
        USE_SEGMENT=true
        USE_DEPTH=true
    fi
    if [ "$3" = style ] ; then
        USE_SEGMENT=true
        USE_STYLE=true
        if [ $# -ge 4 ]; then
            STYLE_NAME=$4
        fi
    fi
fi

# enable Exposure compensation
EXPOSURE_GS_ARG="--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp "
DENSIFY_GS_ARG="--densify_grad_threshold 0.0001 --percent_dense 0.015"
LR_GS_ARG="--position_lr_init 0.000016 --scaling_lr 0.001"

DAY_NIGHT_SCENE_LINE="--white_background"
variable="night"
if echo "$SESSION" | grep -q "$variable"; then
    echo "====================== PROCESS NIGHT ======================"
    echo "The session is in night, where the L1 loss will be small, since the scene is dark. so we increase lambda_dssim."
    DAY_NIGHT_SCENE_LINE="--lambda_dssim 0.5"
    # decrease the threshold, since night scene has less texture
    DENSIFY_GS_ARG="--densify_grad_threshold 0.00005 --percent_dense 0.015"
fi

echo "Process session : " ${SESSION} ", use depth: " ${USE_DEPTH} ", use segment: " ${USE_SEGMENT} ", use style: " ${USE_STYLE} ${STYLE_NAME}

export PYTHONPATH="$PYTHONPATH:submodules/libs"
OUTPUT_FOLDER=output
DEPTH_GS_ARG=""
if [ "$USE_DEPTH" = true ] ; then
  echo "====================== PROCESS DEPTH ESTIMATION ======================"
  echo "for some scenes (e.g., the DeepBlending scenes) it improves quality significantly; for others it either makes a small difference or can even be worse."
  DEPTH_GS_ARG=" --depths ${MAP_FOLDER}/${SESSION}/depths "
  OUTPUT_FOLDER=${OUTPUT_FOLDER}_dep
  echo ${DEPTH_GS_ARG}

  DEPTH_DIRECTORY=${MAP_FOLDER}/${SESSION}/depths
  if [ ! -d "$DEPTH_DIRECTORY" ]; then
      echo "$DEPTH_DIRECTORY does not exist."
      echo "====================== PROCESS DEPTH ESTIMATION ======================"
      # prcocess each folder separately
      cd ${MAP_FOLDER}/Depth-Anything-V2
      for IMAGE_FOLDER in ${MAP_FOLDER}/${SESSION}/dense/images/*/; do
          PATH_SPLIT=(${IMAGE_FOLDER//// })
          FOLDER_NAME=(${PATH_SPLIT[-1]})
          echo "process subfolder " ${IMAGE_FOLDER} ${FOLDER_NAME}
          python run.py --encoder vitl --pred-only --grayscale \
          --img-path ${MAP_FOLDER}/${SESSION}/dense/images/${FOLDER_NAME} \
          --outdir ${MAP_FOLDER}/${SESSION}/depths/${FOLDER_NAME}
      done

      # if has images directly in the images folder, process those
      NUM_IMAGES=$(find ${MAP_FOLDER}/${SESSION}/dense/images/ -maxdepth 1 -type f | wc -l)
      if ((${NUM_IMAGES} > 0)); then
          echo "has images directly in the images folder"
          python run.py --encoder vitl --pred-only --grayscale \
          --img-path ${MAP_FOLDER}/${SESSION}/dense/images \
          --outdir ${MAP_FOLDER}/${SESSION}/depths
      fi

      cd ${MAP_FOLDER}/GaussianSplatting
      python utils/make_depth_scale.py \
      --base_dir ${MAP_FOLDER}/${SESSION}/dense \
      --depths_dir ${MAP_FOLDER}/${SESSION}/depths
  fi
fi

IMAGES_GS_ARG="--images images"
if [ "$USE_SEGMENT" = true ] ; then
    echo "====================== LOAD SEGMENTATION ======================"
    SEGMENT_DIRECTORY=${MAP_FOLDER}/${SESSION}/dense/segment
    if [ ! -d "$SEGMENT_DIRECTORY" ]; then
        echo "$SEGMENT_DIRECTORY does not exist."
        echo "====================== PROCESS SEGMENTATION ======================"
        python dm/segment/segment_dino.py \
        --input ${MAP_FOLDER}/${SESSION}/dense/images \
        --output ${SEGMENT_DIRECTORY}
    fi
    IMAGES_GS_ARG="--images segment"
    OUTPUT_FOLDER=${OUTPUT_FOLDER}_seg
fi

if [ "$USE_STYLE" = true ] ; then
    echo "====================== LOAD STYLE ======================"
    STYLE_DIRECTORY=${MAP_FOLDER}/${SESSION}/dense/style_${STYLE_NAME}
    if [ ! -d "$STYLE_DIRECTORY" ]; then
        echo "$STYLE_DIRECTORY does not exist."
        echo "====================== PROCESS STYLE ======================"
        python dm/style_transfer/dm_transfer_images.py \
        --input_path ${MAP_FOLDER}/${SESSION}/dense \
        --output_path ${MAP_FOLDER}/${SESSION}/dense/style_${STYLE_NAME} \
        --style_path /mnt/data/yeliu/models/style_models/${STYLE_NAME}
    fi
    IMAGES_GS_ARG="--images style_${STYLE_NAME}"
    OUTPUT_FOLDER=${OUTPUT_FOLDER}_${STYLE_NAME}
fi

echo "====================== PROCESS GAUSSSIAN SPLATTING ======================"
echo "====================== ${OUTPUT_FOLDER} ======================"
echo "====================== ${DEPTH_GS_ARG} ${IMAGES_GS_ARG} ======================"
echo "To do the full training routine and avoid running out of memory, you can increase the --densify_grad_threshold (0.0002), --densification_interval (100) or reduce the value of --densify_until_iter (15_000)."
echo "use --start_checkpoint ${MAP_FOLDER}/${SESSION}/${OUTPUT_FOLDER} to continue training"
python train.py ${DEPTH_GS_ARG} ${IMAGES_GS_ARG} ${DAY_NIGHT_SCENE_LINE} ${EXPOSURE_GS_ARG} \
--source_path ${MAP_FOLDER}/${SESSION}/dense \
--model_path  ${MAP_FOLDER}/${SESSION}/${OUTPUT_FOLDER} \
${LR_GS_ARG} \
--iterations 80_000 \
--resolution -1 \
--save_iterations 7000 20000 50000 80000 \
--checkpoint_iterations 80000 \
--densify_until_iter 15_000 ${DENSIFY_GS_ARG} \
--optimizer_type sparse_adam \
--skip_interval 1 \
--disable_viewer

# python train.py ${DEPTH_GS_ARG} ${IMAGES_GS_ARG} ${DAY_NIGHT_SCENE_LINE} ${EXPOSURE_GS_ARG} \
# --source_path ${MAP_FOLDER}/${SESSION}/dense \
# --model_path  ${MAP_FOLDER}/${SESSION}/${OUTPUT_FOLDER} \
# --position_lr_init 0.000016 --scaling_lr 0.003 \
# --iterations 120_000 \
# --resolution 1 \
# --save_iterations 120000 140000 \
# --checkpoint_iterations 140000 \
# --start_checkpoint ${MAP_FOLDER}/${SESSION}/${OUTPUT_FOLDER}/chkpnt80000.pth \
# --densify_until_iter 120_000 --densify_grad_threshold 0.0001 \
# --optimizer_type sparse_adam \
# --disable_viewer \
# --dynamic_memory

echo "====================== DONE ======================"
