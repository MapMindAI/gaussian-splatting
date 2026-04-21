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

import urllib


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation network")
    parser.add_argument("--input", help="test image path", type=str)
    parser.add_argument("--output", help="test image path", type=str)
    parser.add_argument(
        "--weights",
        help="cityscape pretrained weights",
        default="/mnt/data/yeliu/models/ppliteset_pp2torch_cityscape_pretrained.pth",
        type=str,
    )
    args = parser.parse_args()

    return args


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


def render_segmentation(image, segmentation_logits, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    render_image = 0.3 * segmentation_values + 0.7 * image
    return Image.fromarray(render_image.astype(np.uint8))


# https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation/utils/colormaps.py
# 4 tree 17 plant
# 2 sky 12 115 people 21 26 60 128 water
# 20 car 82 truck 116 127 bike 80 bus
def ObjectEncode(labelmap, label_set=[2, 12, 115, 20, 80, 82, 116, 127]):
    labelmap = labelmap.astype("int")
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1]), dtype=np.uint8)
    for label in np.unique(labelmap):
        if not label in label_set:
            continue
        labelmap_rgb = labelmap_rgb + (labelmap == label)[:, :] * 1
    return 1 - labelmap_rgb


def subdivide_image_and_process(model, array_full):
    half_shape = [int(array_full.shape[0] / 2), int(array_full.shape[1] / 2)]
    image_rois = [
        [0, half_shape[0], 0, half_shape[1]],
        [half_shape[0], array_full.shape[0], 0, half_shape[1]],
        [0, half_shape[0], half_shape[1], array_full.shape[1]],
        [half_shape[0], array_full.shape[0], half_shape[1], array_full.shape[1]],
    ]
    segmentation_logits = np.zeros((array_full.shape[0], array_full.shape[1]), dtype=np.uint8)
    for image_roi in image_rois:
        array = array_full[image_roi[0] : image_roi[1], image_roi[2] : image_roi[3], :]
        segmentation_logits_sub = inference_segmentor(model, array)[0]
        segmentation_logits[image_roi[0] : image_roi[1], image_roi[2] : image_roi[3]] = segmentation_logits_sub
    return segmentation_logits


WARNED = False


def run_image(model, image_path, save_path):
    image = Image.open(image_path).convert("RGB")

    array = np.array(image)[:, :, ::-1]  # BGR
    if array.shape[0] > 2000:
        global WARNED
        if not WARNED:
            print("[ INFO ] image too large, subdivide the image and process segmentation")
            WARNED = True
        segmentation_logits = subdivide_image_and_process(model, array)
    else:
        segmentation_logits = np.array(inference_segmentor(model, array)[0])

    # create masked image
    # add mask to images and save
    image_rgba = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)
    image_rgba[:, :, :3] = array[:, :, ::-1]
    image_rgba[:, :, 3] = 255 * ObjectEncode(segmentation_logits).astype(np.uint8)
    img_pil = Image.fromarray(image_rgba, "RGBA")
    img_pil.save(save_path + ".png")

    # render the debug image
    segmented_image = render_segmentation(array, segmentation_logits, "ade20k")
    segmented_image.save(save_path + "_seg.jpg")

    # save the segmentation_logits
    img_label = Image.fromarray(segmentation_logits.astype(np.uint8), "L")
    img_label.save(save_path + "_label.png")


"""
SESSION=wuxi_20241114
python dm/segment/segment_dino.py \
--input /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/images \
--output  /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/segment
"""
if __name__ == "__main__":
    args = parse_args()

    CONFIG_URL = "/mnt/data/yeliu/models/dino2/dinov2_vitg14_ade20k_m2f_config.py"
    CHECKPOINT_URL = "/mnt/data/yeliu/models/dino2/dinov2_vitg14_ade20k_m2f.pth"

    print("===> load", CHECKPOINT_URL)

    # cfg_str = load_config_from_url(CONFIG_URL)
    with open(CONFIG_URL, "r") as file:
        cfg_str = file.read()
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    model = init_segmentor(cfg)
    load_checkpoint(model, CHECKPOINT_URL, map_location="cuda")
    model.cuda()
    model.eval()

    targets = []
    targets += glob.glob(args.input + "/*.jpg")
    targets += glob.glob(args.input + "/*.JPG")
    targets += glob.glob(args.input + "/*.png")
    targets += glob.glob(args.input + "/**/*.jpg", recursive=False)
    targets += glob.glob(args.input + "/**/*.JPG", recursive=False)
    targets += glob.glob(args.input + "/**/*.png", recursive=False)

    print("===> " + args.input + " load " + str(len(targets)) + " images")
    os.makedirs(args.output, exist_ok=True)

    cnt = 1
    for image_path in targets:
        print(f"Processing {cnt}/{len(targets)} {image_path}...")
        cnt += 1

        image_sub_path = image_path[len(args.input) : -4]
        save_path = args.output + "/" + image_sub_path

        save_folder = "/".join(save_path.split("/")[:-1])
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        run_image(model, image_path, save_path)


"""
ADE20K_CLASS_NAMES = [
    "",
0     "wall",
1     "building;edifice",
2     "sky",
3     "floor;flooring",
4     "tree",
5     "ceiling",
6     "road;route",
7     "bed",
8     "windowpane;window",
9     "grass",
10    "cabinet",
11    "sidewalk;pavement",
12    "person;individual;someone;somebody;mortal;soul",
13    "earth;ground",
14    "door;double;door",
15    "table",
16    "mountain;mount",
17    "plant;flora;plant;life",
18    "curtain;drape;drapery;mantle;pall",
19    "chair",
20    "car;auto;automobile;machine;motorcar",
    "water",
    "painting;picture",
    "sofa;couch;lounge",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug;carpet;carpeting",
    "field",
30   "armchair",
    "seat",
    "fence;fencing",
    "desk",
    "rock;stone",
    "wardrobe;closet;press",
    "lamp",
    "bathtub;bathing;tub;bath;tub",
    "railing;rail",
    "cushion",
40    "base;pedestal;stand",
    "box",
    "column;pillar",
    "signboard;sign",
    "chest;of;drawers;chest;bureau;dresser",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace;hearth;open;fireplace",
50    "refrigerator;icebox",
    "grandstand;covered;stand",
    "path",
    "stairs;steps",
    "runway",
    "case;display;case;showcase;vitrine",
    "pool;table;billiard;table;snooker;table",
    "pillow",
    "screen;door;screen",
    "stairway;staircase",
60    "river",
    "bridge;span",
    "bookcase",
    "blind;screen",
    "coffee;table;cocktail;table",
    "toilet;can;commode;crapper;pot;potty;stool;throne",
    "flower",
    "book",
    "hill",
    "bench",
70    "countertop",
    "stove;kitchen;stove;range;kitchen;range;cooking;stove",
    "palm;palm;tree",
    "kitchen;island",
    "computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system",
    "swivel;chair",
    "boat",
    "bar",
    "arcade;machine",
    "hovel;hut;hutch;shack;shanty",
80    "bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle",
    "towel",
    "light;light;source",
    "truck;motortruck",
    "tower",
    "chandelier;pendant;pendent",
    "awning;sunshade;sunblind",
    "streetlight;street;lamp",
    "booth;cubicle;stall;kiosk",
    "television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box",
90    "airplane;aeroplane;plane",
    "dirt;track",
    "apparel;wearing;apparel;dress;clothes",
    "pole",
    "land;ground;soil",
    "bannister;banister;balustrade;balusters;handrail",
    "escalator;moving;staircase;moving;stairway",
    "ottoman;pouf;pouffe;puff;hassock",
    "bottle",
    "buffet;counter;sideboard",
100    "poster;posting;placard;notice;bill;card",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer;belt;conveyor;belt;conveyer;conveyor;transporter",
    "canopy",
    "washer;automatic;washer;washing;machine",
    "plaything;toy",
    "swimming;pool;swimming;bath;natatorium",
110    "stool",
    "barrel;cask",
    "basket;handbasket",
    "waterfall;falls",
    "tent;collapsible;shelter",
    "bag",
    "minibike;motorbike",
    "cradle",
    "oven",
    "ball",
120    "food;solid;food",
    "step;stair",
    "tank;storage;tank",
    "trade;name;brand;name;brand;marque",
    "microwave;microwave;oven",
    "pot;flowerpot",
    "animal;animate;being;beast;brute;creature;fauna",
    "bicycle;bike;wheel;cycle",
    "lake",
    "dishwasher;dish;washer;dishwashing;machine",
130    "screen;silver;screen;projection;screen",
    "blanket;cover",
    "sculpture",
    "hood;exhaust;hood",
    "sconce",
    "vase",
    "traffic;light;traffic;signal;stoplight",
    "tray",
    "ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin",
    "fan",
140    "pier;wharf;wharfage;dock",
    "crt;screen",
    "plate",
    "monitor;monitoring;device",
    "bulletin;board;notice;board",
    "shower",
    "radiator",
    "glass;drinking;glass",
    "clock",
    "flag",
]
"""
