#!/usr/bin/env bash
set -e


dataset=$1
dataset_out=$2

# check if the model exist
SPARSE_MODEL_PATH=${dataset_out}/reconstruction/sparse/0
if [ -d "$SPARSE_MODEL_PATH" ]; then
    echo "SPARSE MODEL EXIST"
    exit 0
fi

mkdir -p $dataset_out

openMVG_main_SfMInit_ImageListing -i $dataset -o $dataset_out/matches -f 1 -c 7 -P

openMVG_main_ComputeFeatures -i $dataset_out/matches/sfm_data.json -o $dataset_out/matches -m SIFT -p ULTRA -u 1 --numThreads 4

openMVG_main_ListMatchingPairs -i $dataset_out/matches/sfm_data.json -o $dataset_out/matches/matching_pairs.txt -n 20 -G

openMVG_main_ComputeMatches -i $dataset_out/matches/sfm_data.json -o $dataset_out/matches/matches_putative.bin -p $dataset_out/matches/matching_pairs.txt -n HNSWL1

openMVG_main_GeometricFilter -i $dataset_out/matches/sfm_data.json -o $dataset_out/matches/matches_refined.bin -m $dataset_out/matches/matches_putative.bin -g u

# STELLAR GLOBAL INCREMENTAL : GLOBAL may failed and slow. STELLAR can work but points too few. INCREMENTAL has sometimes better, sometime worse
# https://github.com/openMVG/openMVG/pull/2070
openMVG_main_SfM -i $dataset_out/matches/sfm_data.json -M $dataset_out/matches/matches_putative.bin -o $dataset_out/reconstruction -s STELLAR

openMVG_main_openMVGSpherical2Cubic -i $dataset_out/reconstruction/sfm_data.bin -o $dataset_out/reconstruction/images

mkdir -p $dataset_out/reconstruction/sparse/0
openMVG_main_openMVG2Colmap -i $dataset_out/reconstruction/images/sfm_data_perspective.bin -o $dataset_out/reconstruction/sparse/0
