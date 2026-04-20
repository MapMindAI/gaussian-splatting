#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo "PLEASE SET <MAP_FOLDER> <SESSION NAME>"
    exit 1
fi

USE_DEPTH=false
MAP_FOLDER=$1
SESSION=$2

if [ $# -ge 3 ]; then
    if [ "$3" = depth ] ; then
        USE_DEPTH=true
    fi
fi

# enable Exposure compensation
EXPOSURE_GS_ARG="--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp "
DENSIFY_GS_ARG="--densify_grad_threshold 0.00008 --percent_dense 0.015"
LR_GS_ARG="--position_lr_init 0.000016 --scaling_lr 0.001"

DAY_NIGHT_SCENE_LINE="--white_background"
# add night to name if collect in night
variable="night"
if echo "$SESSION" | grep -q "$variable"; then
    echo "====================== PROCESS NIGHT ======================"
    echo "The session is in night, where the L1 loss will be small, since the scene is dark. so we increase lambda_dssim."
    DAY_NIGHT_SCENE_LINE="--lambda_dssim 0.5"
    # decrease the threshold, since night scene has less texture
    DENSIFY_GS_ARG="--densify_grad_threshold 0.00005 --percent_dense 0.015"
fi

TEST_PATH=${MAP_FOLDER}/${SESSION}/images/use_opencv_model/
IF_USE_DENSE=""
if [ -d "$TEST_PATH" ]; then
  IF_USE_DENSE="/dense"
fi

echo "Process session : " ${SESSION} ${IF_USE_DENSE}

export PYTHONPATH="$PYTHONPATH:submodules/libs"
OUTPUT_FOLDER=output

DEPTH_GS_ARG=""
if [ "$USE_DEPTH" = true ] ; then
  echo "====================== PROCESS DEPTH ESTIMATION ======================"
  echo "for some scenes (e.g., the DeepBlending scenes) it improves quality significantly; for others it either makes a small difference or can even be worse."
  DEPTH_GS_ARG=" --depths ${MAP_FOLDER}/${SESSION}/depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01"
  OUTPUT_FOLDER=${OUTPUT_FOLDER}_dep
  echo ${DEPTH_GS_ARG}

  DEPTH_DIRECTORY=${MAP_FOLDER}/${SESSION}/depths
  if [ ! -d "$DEPTH_DIRECTORY" ]; then
      echo "$DEPTH_DIRECTORY does not exist."
      echo "====================== PROCESS DEPTH ESTIMATION ======================"
      # prcocess each folder separately
      cd /mnt/data/yeliu/gaussian_splatting/Depth-Anything-V2
      for IMAGE_FOLDER in ${MAP_FOLDER}/${SESSION}${IF_USE_DENSE}/images/*/; do
          PATH_SPLIT=(${IMAGE_FOLDER//// })
          FOLDER_NAME=(${PATH_SPLIT[-1]})
          echo "process subfolder " ${IMAGE_FOLDER} ${FOLDER_NAME}
          python run.py --encoder vitl --pred-only --grayscale \
          --img-path ${MAP_FOLDER}${SESSION}${IF_USE_DENSE}/images/${FOLDER_NAME} \
          --outdir ${MAP_FOLDER}/${SESSION}/depths/${FOLDER_NAME}
      done

      # if has images directly in the images folder, process those
      NUM_IMAGES=$(find ${MAP_FOLDER}/${SESSION}${IF_USE_DENSE}/images/ -maxdepth 1 -type f | wc -l)
      if ((${NUM_IMAGES} > 0)); then
          echo "has images directly in the images folder"
          python run.py --encoder vitl --pred-only --grayscale \
          --img-path ${MAP_FOLDER}/${SESSION}${IF_USE_DENSE}/images \
          --outdir ${MAP_FOLDER}/${SESSION}/depths
      fi

      cd /mnt/data/yeliu/Dev/GaussianSplatting
      python utils/make_depth_scale.py \
      --base_dir ${MAP_FOLDER}/${SESSION}${IF_USE_DENSE} \
      --depths_dir ${MAP_FOLDER}/${SESSION}/depths \
      --model_type txt
  fi
fi

NUM_IMAGES=$(find ${MAP_FOLDER}/${SESSION}/images -type f | wc -l)
ITERATIONS=$((NUM_IMAGES * 50 / 1000 * 1000))

echo "====================== PROCESS GAUSSSIAN SPLATTING ======================"
echo "====================== ${IF_USE_DENSE} ${OUTPUT_FOLDER} ${ITERATIONS} ======================"
echo "To do the full training routine and avoid running out of memory, you can increase the --densify_grad_threshold (0.0002), --densification_interval (100) or reduce the value of --densify_until_iter (15_000)."
echo "use --start_checkpoint ${MAP_FOLDER}/${SESSION}/${OUTPUT_FOLDER} to continue training"
cd /mnt/data/yeliu/Dev/GaussianSplatting
python train.py ${DAY_NIGHT_SCENE_LINE} ${EXPOSURE_GS_ARG} ${DEPTH_GS_ARG} \
--source_path ${MAP_FOLDER}/${SESSION}${IF_USE_DENSE} \
--model_path  ${MAP_FOLDER}/${SESSION}/${OUTPUT_FOLDER} \
${LR_GS_ARG} \
--iterations ${ITERATIONS} \
--resolution -1 \
--save_iterations 10000 40000 80000 \
--checkpoint_iterations 80000 \
--densify_until_iter $((ITERATIONS / 2)) ${DENSIFY_GS_ARG} \
--optimizer_type sparse_adam \
--skip_interval 1 \
--disable_viewer

echo "====================== DONE ======================"
