import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch
import re
import glob
import os
import numpy as np
import cv2
from torchvision import transforms
from transformer_net import TransformerNet
from vgg import Vgg16
from pathlib import Path


def load_style_model(model_path, device=torch.device("cuda")):
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r"in\d+\.running_(mean|var)$", k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        return style_model


def stylize_blocks(style_model, content_image, divide=2, device=torch.device("cuda")):
    if divide <= 1:
        return stylize(style_model, content_image, device)

    half_shape = [int(content_image.shape[0] / 2), int(content_image.shape[1] / 2)]
    border = 50
    image_rois = [
        [0, half_shape[0], 0, half_shape[1]],
        [half_shape[0], content_image.shape[0], 0, half_shape[1]],
        [0, half_shape[0], half_shape[1], content_image.shape[1]],
        [half_shape[0], content_image.shape[0], half_shape[1], content_image.shape[1]],
    ]
    offsets = [
        [0, border, 0, border],
        [-border, 0, 0, border],
        [0, border, -border, 0],
        [-border, 0, -border, 0],
    ]
    current_divide = divide - 1

    style_image = np.zeros((content_image.shape[0], content_image.shape[1], 3), dtype=np.uint8)
    for i in range(len(offsets)):
        image_roi = image_rois[i]
        offset = offsets[i]
        sub_image = content_image[
            image_roi[0] + offset[0] : image_roi[1] + offset[1],
            image_roi[2] + offset[2] : image_roi[3] + offset[3],
            :,
        ]
        sub_style_image = stylize_blocks(style_model, sub_image, current_divide, device)
        style_image[image_roi[0] : image_roi[1], image_roi[2] : image_roi[3], :] = sub_style_image[
            -offset[0] : sub_style_image.shape[0] - offset[1],
            -offset[2] : sub_style_image.shape[1] - offset[3],
        ]
    return style_image


def stylize(style_model, content_image, device=torch.device("cuda")):
    content_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    # print(content_image.shape)
    with torch.no_grad():
        output = style_model(content_image).cpu()

        img = output[0].clone().clamp(0, 255).numpy()
        img_array = img.transpose(1, 2, 0).astype("uint8")
        img_array = cv2.resize(img_array, (content_image.shape[3], content_image.shape[2]))
        return img_array


# https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation/utils/colormaps.py
# 4 tree 17 plant
# 2 sky 12 115 people 21 26 60 128 water
def MaskPeopleSky(labelmap, label_set=[2, 12, 115]):
    labelmap = labelmap.astype("int")
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1]), dtype=np.uint8)
    for label in np.unique(labelmap):
        if not label in label_set:
            continue
        labelmap_rgb = labelmap_rgb + (labelmap == label)[:, :] * 1
    return 1 - labelmap_rgb


# https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation/utils/colormaps.py
# 4 tree 17 plant 9 grass
# 2 sky 12 115 people 21 26 60 128 water
def segmented_image_style(content_image, segment_image, stylized_image, update_labels=[4, 9, 17]):
    segment_image = np.array(segment_image).astype("int")
    result_image = np.array(content_image.copy())

    for label in np.unique(segment_image):
        if not label in update_labels:
            continue
        flags = segment_image == label
        result_image[flags] = stylized_image[flags]

    image_rgba = np.zeros((result_image.shape[0], result_image.shape[1], 4), dtype=np.uint8)
    image_rgba[:, :, :3] = result_image[:, :, :]
    image_rgba[:, :, 3] = 255 * MaskPeopleSky(segment_image).astype(np.uint8)
    return image_rgba


def calculate_image_statistics(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_l, std_l = l.mean(), l.std()
    mean_a, std_a = a.mean(), a.std()
    mean_b, std_b = b.mean(), b.std()
    return {
        "mean_l": mean_l,
        "std_l": std_l,
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
    }


def match_color(image, target_stats_raw, target_ratio=1.0):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_f, a_f, b_f = cv2.split(lab_image)
    f_stats = calculate_image_statistics(image)
    target_stats = dict(target_stats_raw)
    for key in f_stats:
        target_stats[key] = (1.0 - target_ratio) * target_stats_raw[key] + target_ratio * f_stats[key]

    l_matched = (l_f - f_stats["mean_l"]) / f_stats["std_l"] * target_stats["std_l"] + target_stats["mean_l"]
    a_matched = (a_f - f_stats["mean_a"]) / f_stats["std_a"] * target_stats["std_a"] + target_stats["mean_a"]
    b_matched = (b_f - f_stats["mean_b"]) / f_stats["std_b"] * target_stats["std_b"] + target_stats["mean_b"]
    l_matched = np.clip(l_matched, 0, 255).astype(np.uint8)
    a_matched = np.clip(a_matched, 0, 255).astype(np.uint8)
    b_matched = np.clip(b_matched, 0, 255).astype(np.uint8)
    lab_matched = cv2.merge([l_matched, a_matched, b_matched]).astype(np.uint8)
    matched_image = cv2.cvtColor(lab_matched, cv2.COLOR_LAB2BGR)
    return matched_image


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation network")
    parser.add_argument("--input_path", help="test image path", type=str)
    parser.add_argument("--output_path", help="test image path", type=str)
    parser.add_argument("--style_path", help="style model path", type=str)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("--segment", default="segment", type=str)
    args = parser.parse_args()
    return args


"""
SESSION=wuxi_20241114
python dm/style_transfer/dm_transfer_images.py \
--input_path /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense \
--output_path /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/style \
--style_path /mnt/data/yeliu/models/style_models/autum
"""
if __name__ == "__main__":
    args = parse_args()
    style_name = args.style_path.split("/")[-1]
    update_labels = [4, 9, 17]
    if "winter" in style_name:
        update_labels.append(1)
    print(f"[ INFO ] load style {style_name}, update labels {update_labels}")

    style_model_paths = glob.glob(args.style_path + "/*.pth")
    if len(style_model_paths) != 1:
        print(f"[ ERROR ] found {len(style_model_paths)} style models.")
        exit(0)

    print(f"[ INFO ] load style model from {style_model_paths[0]}")
    style_model = load_style_model(style_model_paths[0])

    style_image_paths = glob.glob(args.style_path + "/*.jpg") + glob.glob(args.style_path + "/*.png")
    if len(style_image_paths) != 1:
        print(f"[ ERROR ] found {len(style_image_paths)} style images.")
        exit(0)

    print(f"[ INFO ] load style image from {style_image_paths[0]}")
    style_base_image = np.array(Image.open(style_image_paths[0]))
    style_base_stats = calculate_image_statistics(style_base_image)
    print(f"[ INFO ] style image statistics : {style_base_stats}")

    images_folder = args.input_path + "/" + args.images
    targets = []
    targets += glob.glob(images_folder + "/*.jpg")
    targets += glob.glob(images_folder + "/*.JPG")
    targets += glob.glob(images_folder + "/*.png")
    targets += glob.glob(images_folder + "/**/*.jpg", recursive=False)
    targets += glob.glob(images_folder + "/**/*.JPG", recursive=False)
    targets += glob.glob(images_folder + "/**/*.png", recursive=False)
    print(f"[ INFO ] load {len(targets)} images.")

    segmentation_folder = args.input_path + "/" + args.segment + "/"
    if not os.path.exists(segmentation_folder):
        print(f"[ ERROR ] {segmentation_folder} not found!")
        exit(0)

    # output folder
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)
    print(f"[ INFO ] save to {output_folder}.")

    divide_level = None

    # process images
    cnt = 0
    for image_path in targets:
        cnt += 1
        image_name = image_path[len(images_folder) :]
        if image_name[0] == "/":
            image_name = image_name[1:]

        print(f"   process {cnt}/{len(targets)} {image_name}")

        content_image = np.array(Image.open(image_path).convert("RGB"))
        seg_image_path = segmentation_folder + image_name[:-4] + "_label.png"
        segment_image = np.array(Image.open(seg_image_path))
        if divide_level is None:
            divide_level = int(np.ceil(np.log2(segment_image.shape[1] / 1000)) + 1)
            print(f"[ INFO ] divide level set to be {divide_level}")

        stylized_image = stylize_blocks(style_model, content_image, divide_level)

        matched_content = match_color(content_image, style_base_stats, 0.5)
        result_image = segmented_image_style(matched_content, segment_image, stylized_image, update_labels)
        # result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        image_save_path = output_folder + "/" + image_name[:-4] + ".png"
        image_name_splits = image_name.split("/")
        if len(image_name_splits) > 1:
            # need to create subfolder
            Path(output_folder + "/" + image_name.split("/")[0]).mkdir(parents=True, exist_ok=True)

        img_pil = Image.fromarray(result_image, "RGBA")
        img_pil.save(image_save_path)

    print("[ INFO ] Done!")
