import os
import glob
import sys
import cv2  # type: ignore
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import json
from typing import Any, Dict, List


MODELS_FOLDER="/mnt/data/yeliu/models/segment_anything"
# pip install git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
def prepare_models(model_names = ["sam_vit_h_4b8939", "sam_vit_l_0b3195", "sam_vit_b_01ec64"]):
    print("====================== PREPARE MODELS ======================")
    for model_name in model_names:
        model_path = MODELS_FOLDER + "/" + model_name + ".pth"
        if os.path.isfile(model_path):
            print("Found " + model_name)
        else:
            os.system("wget -O " + model_path + " https://dl.fbaipublicfiles.com/segment_anything/" + model_name + ".pth")


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def write_opencv_mask_image(masks: List[Dict[str, Any]], path: str) -> None:
    mask_full = None
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        if mask_full is None:
            mask_full = mask * i
        else:
            mask_full += mask * i
    cv2.imwrite(path + ".png", mask_full)
    

"""
SESSION=wuxi_20241114
python dm/segment/segment_anything.py \
--input /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/images \
--output  /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/segment \
--model-type vit_h
"""
if __name__ == "__main__":
    prepare_models()
    output_mode = "binary_mask"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Path to the directory where masks will be output. Output will be either a folder "
            "of PNGs per image or a single json with COCO-style masks."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )
    args = parser.parse_args(sys.argv[1:])
    
    checkpoint_path = MODELS_FOLDER + "/sam_vit_b_01ec64.pth"
    if args.model_type.find("vit_h") != -1:
        checkpoint_path = MODELS_FOLDER + "/sam_vit_h_4b8939.pth"
    elif args.model_type.find("vit_l") != -1:
        checkpoint_path = MODELS_FOLDER + "/sam_vit_l_0b3195.pth"
    
    print("load " + checkpoint_path)
    sam = sam_model_registry[args.model_type](checkpoint=checkpoint_path)
    _ = sam.to(device="cuda")
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
    
    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]
    print("load " + str(len(targets)) + " images")
    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        #os.makedirs(save_base, exist_ok=False)
        #write_masks_to_folder(masks, save_base)
        write_opencv_mask_image(masks, save_base)
    print("Done!")
