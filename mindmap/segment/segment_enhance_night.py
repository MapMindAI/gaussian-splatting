
import sys
import math
import os
import glob
import numpy as np
import itertools
from functools import partial
import urllib
from pathlib import Path
import threading

from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F

import argparse
import cv2

import urllib

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--segment',
                        help='test image path',
                        type=str)
    parser.add_argument('--input',
                        help='test image path',
                        type=str)
    parser.add_argument('--output',
                        help='test image path',
                        type=str)
    args = parser.parse_args()
    return args


def apply_gamma_correction(img_array, gamma=0.5):
    """
    Apply gamma correction to enhance dark areas of the image.
    :param image: PIL Image object
    :param gamma: Gamma value for correction (e.g., 2.2 for brighter images)
    :return: Gamma-corrected image
    """
    # Normalize to range [0, 1]
    img_normalized = img_array / 255.0
    # Apply gamma correction
    img_corrected = np.power(img_normalized, gamma)
    # Scale back to range [0, 255]
    img_corrected = (img_corrected * 255).astype(np.uint8)
    # Convert back to PIL Image
    return img_corrected


def smooth_dark_enhancement(image, gamma_min=0.5, gamma_max=1.0, alpha=2.0, keep_threshold=100.0, blur_strength=15):
    """
    Smoothly enhance dark regions using adaptive gamma correction.
    
    :param image: PIL Image object.
    :param gamma_min: Minimum gamma value for darkest regions.
    :param gamma_max: Maximum gamma value for brightest regions (default = no change).
    :param alpha: Controls the smooth transition strength (higher = sharper transitions).
    :return: Enhanced PIL Image.
    """
    # Convert image to NumPy array
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    # a = (255.0 - min_bright) / 255.0
    # b = 50.0 / 255.0
    # enhanced_img = a * img_array + b
    
    # b = 1.0
    # a = (keep_threshold / 255.0) / np.log10(keep_threshold / 255.0 + b)
    # enhanced_img = a * np.log10(img_array + b)
    # enhanced_img[img_array > keep_threshold] = img_array[img_array > keep_threshold]
    
    # Apply bilateral filter or Gaussian blur to denoise dark regions
    img_denoised = cv2.bilateralFilter((img_array * 255).astype(np.uint8), 
                                       d=blur_strength, 
                                       sigmaColor=75, 
                                       sigmaSpace=75)
    enhanced_img = img_denoised / 255.0  # Back to [0, 1]
    enhanced_img[img_array > keep_threshold] = img_array[img_array > keep_threshold]
    
    # Rescale to [0, 255] and convert back to uint8
    enhanced_img = (enhanced_img * 255).clip(0, 255).astype(np.uint8)
    return enhanced_img


def edge_preserving_denoise(img_array, blur_strength=10, edge_preserve_weight=0.75):
    """
    Apply edge-aware denoising using a combination of bilateral filter and edge preservation.
    
    :param image: PIL Image object.
    :param blur_strength: Strength of the bilateral filter (higher = more smoothing).
    :param edge_preserve_weight: Weight for edge preservation (0-1, higher preserves more edges).
    :return: Denoised PIL Image.
    """    
    # Compute edges using a Sobel filter
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 5, 150)  # Detect edges using Canny
    
    # Dilate the edges to emphasize them
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Apply bilateral filter for denoising
    img_denoised = cv2.bilateralFilter(img_array, 
                                       d=blur_strength, 
                                       sigmaColor=75, 
                                       sigmaSpace=75)
    
    # Preserve edges by blending original and denoised images
    mask = (edges_dilated > 0).astype(np.float32)[..., np.newaxis]
    enhanced_img = (1 - edge_preserve_weight) * img_denoised + edge_preserve_weight * img_array * mask + \
                   edge_preserve_weight * (1 - mask) * img_denoised
    
    # Clip values to valid range and convert back to uint8
    enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
    return enhanced_img


def run_image(image_path, save_path):
    image = Image.open(image_path)
#     enhancer = ImageEnhance.Brightness(image)
#     bright_image = enhancer.enhance(1.5)  # Increase brightness by 1.5x
    array = np.array(image)
    
    # create masked image
    # add mask to images and save
    image_rgba = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)
    
    # image_rgba[:, :, :3] = array[:, :, :3]
    image_rgba[:, :, :3] = edge_preserving_denoise(array[:, :, :3])
    image_rgba[:, :, 3] = array[:, :, 3]
    
    
    img_pil = Image.fromarray(image_rgba, 'RGBA')
    img_pil.save(save_path + ".png")

"""
SESSION=nanchansi_night
python dm/segment/segment_enhance_night.py \
--input /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/images \
--segment /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/segment \
--output /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/segment
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
    
    print("===>  load " + str(len(targets)) + " images")
    os.makedirs(args.output, exist_ok=True)
    
    def process_thread(thread_id, start, end):
        length = end - start
        for i in range(length):
            cnt = start + i
            image_path = targets[cnt]
            print(f"Processing {thread_id} {i}/{length} {image_path}...")

            image_sub_path = image_path[len(args.input):-4]
            save_path = args.output + "/" + image_sub_path
            segment_path = args.segment + "/" + image_sub_path + ".png"

            save_folder = '/'.join(save_path.split('/')[:-1])
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            run_image(segment_path, save_path)
    
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
