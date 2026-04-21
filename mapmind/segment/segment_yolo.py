import os
import glob
import sys
import cv2
import argparse
import json
from typing import Any, Dict, List
from ultralytics import YOLO
from ultralytics.engine.results import Masks


# https://github.com/ultralytics/ultralytics
# pip install ultralytics

MODELS_PATH = "/mnt/data/yeliu/models/yolo11n-seg.pt"


# https://github.com/ultralytics/ultralytics/issues/14357
def write_opencv_mask_image(results, path: str) -> None:
    # Convert masks to grayscale
    mask_full = None
    for n, result in enumerate(results):
        print(n)
        result.save(path + str(n) + ".jpg")
        # if result.masks is None:
        #     continue
        # for i, mask in enumerate(result.masks):
        #     mask_tmp = mask.data.detach().cpu().numpy()
        #     print(i, mask_tmp.data)
        #     # grayscale_mask = (mask * 255).astype('uint8')
        #     # cv2.imwrite("grayscale_mask.png", grayscale_mask)
        #     if mask_full is None:
        #         mask_full = (mask_tmp * i).astype('uint8')
        #     else:
        #         mask_full += (mask_tmp * i).astype('uint8')

    # cv2.imwrite(path + ".png", mask_full)


"""
SESSION=wuxi_20241114
python dm/segment/segment_yolo.py \
--input /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/images \
--output  /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/segment
"""
if __name__ == "__main__":
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
    args = parser.parse_args(sys.argv[1:])

    # Load a model
    model = YOLO(MODELS_PATH)  # load a pretrained model (recommended for training)
    model.to(device="cuda")

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))]
        targets = [os.path.join(args.input, f) for f in targets]
    print("load " + str(len(targets)) + " images")
    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")

        # masks = generator.generate(image)
        results = model(t)  # predict on an image

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        write_opencv_mask_image(results, save_base)
        # break
    print("Done!")
