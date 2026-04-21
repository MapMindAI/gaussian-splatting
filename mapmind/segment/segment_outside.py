import sys

REPO_PATH = "/mnt/data/yeliu/gaussian_splatting/GaussianSplatting"  # Specify a local path to the repository (or use installed package instead)
sys.path.append(REPO_PATH)

from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)

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
import cv2

import urllib


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation network")
    parser.add_argument("--model", help="model path", type=str)
    parser.add_argument("--segment", help="test image path", type=str)
    parser.add_argument("--output", help="test image path", type=str)
    args = parser.parse_args()

    return args


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def run_image(points, cam_range, image_path, extr, intr, save_path):
    debug = False

    # project the points to image
    R = qvec2rotmat(extr.qvec)
    T = np.array(extr.tvec).reshape([3, 1])
    if intr.model == "SIMPLE_PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[0]
    elif intr.model == "PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
    else:
        assert (
            False
        ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

    image = Image.open(image_path)
    array = np.array(image)
    cx = array.shape[1] * 0.5
    cy = array.shape[0] * 0.5

    intrin_mat = np.array([[focal_length_x, 0, cx], [0, focal_length_y, cy], [0, 0, 1]])

    # project points to image
    points_cam = np.dot(R, points) + T
    points_mask = (points_cam[2] > 0) & (points_cam[2] < cam_range)
    points_cam = points_cam[:, points_mask]

    points_cam = points_cam / points_cam[2, :]
    points_cam = np.dot(intrin_mat, points_cam)
    points_mask_x = (points_cam[0] > 0) & (points_cam[0] < array.shape[1])
    points_mask_y = (points_cam[1] > 0) & (points_cam[1] < array.shape[0])
    points_cam = points_cam[:, points_mask_x & points_mask_y]
    points_cam_tr = np.transpose(points_cam[0:2, :]).astype(int)

    hull = cv2.convexHull(points_cam_tr)
    close_mask = np.zeros((array.shape[0], array.shape[1]), dtype=np.uint8)
    cv2.fillPoly(close_mask, [hull], color=255)

    close_mask[array[:, :, 3] == 0] = 0

    if debug:
        rgb = cv2.imread(image_path)
        print(points_cam.shape)
        for i in range(points_cam.shape[1]):
            point = points_cam[:, i]
            cv2.circle(rgb, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

    # create masked image
    # add mask to images and save
    image_rgba = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)

    if debug:
        image_rgba[:, :, :3] = rgb
        image_rgba[:, :, 3] = 255
    else:
        image_rgba[:, :, :3] = array[:, :, :3]
        image_rgba[:, :, 3] = close_mask

    img_pil = Image.fromarray(image_rgba, "RGBA")
    img_pil.save(save_path + ".png")


"""
SESSION=nanchansi_day
python dm/segment/segment_outside.py \
--model /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/sparse \
--segment /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/segment \
--output /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense/outside_masked
"""
if __name__ == "__main__":
    args = parse_args()
    range_factor = 1.0

    # read the model
    print("==> Read colmap from " + args.model)
    bin_path = os.path.join(args.model, "points3D.bin")
    xyz, _, errors = read_points3D_binary(bin_path)
    print("===> Load points :", xyz.shape)
    # filter points by error
    error_threshold = 1
    xyz = xyz[errors.flatten() < error_threshold]

    xyz = xyz.transpose()
    print("===> Filtered points :", xyz.shape)

    cameras_extrinsic_file = os.path.join(args.model, "images.bin")
    cameras_intrinsic_file = os.path.join(args.model, "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    targets = [cam_id for cam_id in cam_extrinsics]
    targets = sorted(targets)

    # compute camera range
    positions = []
    for cam_id in cam_extrinsics:
        extr = cam_extrinsics[cam_id]
        R = np.transpose(qvec2rotmat(extr.qvec))
        pos = -np.dot(R, np.array(extr.tvec).reshape([3, 1]))
        positions.append(pos.reshape(3))
    positions = np.array(positions)
    cam_range = range_factor * np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
    print("===>  include range", cam_range)

    print("===>  load " + str(len(targets)) + " images")
    os.makedirs(args.output, exist_ok=True)

    def process_thread(thread_id, start, end):
        length = end - start
        for i in range(length):
            cnt = start + i
            cam_id = targets[cnt]
            camera_extr = cam_extrinsics[cam_id]
            camera_intr = cam_intrinsics[camera_extr.camera_id]

            image_name = camera_extr.name
            print(f"Processing {thread_id} {i}/{length} {image_name}...")

            image_sub_path = image_name[:-4]
            save_path = args.output + "/" + image_sub_path
            segment_path = args.segment + "/" + image_sub_path + ".png"

            save_folder = "/".join(save_path.split("/")[:-1])
            Path(save_folder).mkdir(parents=True, exist_ok=True)

            run_image(xyz, cam_range, segment_path, camera_extr, camera_intr, save_path)

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
