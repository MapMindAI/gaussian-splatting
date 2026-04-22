# https://github.com/facebookresearch/dinov2
# https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb
# https://dinov2.metademolab.com/demos?category=segmentation
import sys

REPO_PATH = "/mnt/data/yeliu/Dev/dinov2"  # Specify a local path to the repository (or use installed package instead)
sys.path.append(REPO_PATH)

import math
import os
import glob
import numpy as np
import itertools
from functools import partial
import urllib
from pathlib import Path
import threading

from PIL import Image
import torch
import torch.nn.functional as F

import argparse
import dinov2.eval.segmentation.models
import dinov2.eval.segmentation_m2f.models.segmentors
import dinov2.eval.segmentation.utils.colormaps as colormaps
from mmseg.apis import init_segmentor, inference_segmentor

import mmcv
from mmcv.runner import load_checkpoint

DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}
PROCESS_SKY = False

import urllib


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation network")
    parser.add_argument("--input", help="test image path", type=str)
    parser.add_argument("--output", help="test image path", type=str)
    args = parser.parse_args()

    return args


# https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation/utils/colormaps.py
# 4 tree 17 plant
# 2 sky 12 115 people 21 26 60 128 water
# 20 car 82 truck 116 127 bike 80 bus
def ObjectEncode(labelmap, label_set=[12, 115, 20, 80, 82, 116, 127]):
    if not PROCESS_SKY:
        label_set.append(2)
    labelmap = labelmap.astype("int")
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1]), dtype=np.uint8)
    for label in label_set:
        labelmap_rgb = labelmap_rgb + (labelmap == label)[:, :] * 1

    sky_mask = (labelmap == 2)[:, :]
    return 1 - labelmap_rgb, sky_mask


WARNED = False


def run_image(image_path, save_path, sky_color):
    image = Image.open(image_path).convert("RGB")
    array = np.array(image)[:, :, ::-1]  # BGR
    # load image
    segmentation_logits = np.array(Image.open(save_path + "_label.png"))

    # create masked image
    # add mask to images and save
    image_rgba = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)
    alpha, sky_mask = ObjectEncode(segmentation_logits)

    # update sky to white
    if PROCESS_SKY:
        array[sky_mask] = sky_color

    image_rgba[:, :, :3] = array[:, :, ::-1]
    image_rgba[:, :, 3] = 255 * alpha.astype(np.uint8)
    img_pil = Image.fromarray(image_rgba, "RGBA")
    img_pil.save(save_path + ".png")


"""
SESSION=wuxi_12_night_full
python dm/segment/segment_dino_recreate.py \
--input /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/images \
--output  /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/segment
"""
if __name__ == "__main__":
    args = parse_args()

    targets = []
    targets += glob.glob(args.input + "/*.jpg")
    targets += glob.glob(args.input + "/*.JPG")
    targets += glob.glob(args.input + "/*.png")
    targets += glob.glob(args.input + "/**/*.jpg", recursive=False)
    targets += glob.glob(args.input + "/**/*.JPG", recursive=False)
    targets += glob.glob(args.input + "/**/*.png", recursive=False)

    print("===> " + args.input + " load " + str(len(targets)) + " images")
    os.makedirs(args.output, exist_ok=True)
    targets.sort()

    sky_color = [255, 255, 255]
    if "night" in args.input:
        print("processing night")
        sky_color = [0, 0, 0]

    def process_thread(thread_id, start, end):
        length = end - start
        for i in range(length):
            cnt = start + i
            image_path = targets[cnt]
            print(f"Processing {thread_id} {i}/{length} {image_path}...")

            image_sub_path = image_path[len(args.input) : -4]
            save_path = args.output + "/" + image_sub_path

            save_folder = "/".join(save_path.split("/")[:-1])
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            run_image(image_path, save_path, sky_color)

    threads = []

    num_thread = 16
    each_thread_size = int(len(targets) / num_thread)
    for i in range(num_thread):
        start = each_thread_size * i
        end = min(each_thread_size * (i + 1), len(targets))
        t1 = threading.Thread(target=process_thread, args=(i, start, end))
        threads.append(t1)

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
