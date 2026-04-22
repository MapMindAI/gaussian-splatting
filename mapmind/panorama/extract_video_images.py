# https://github.com/adynathos/panorama_to_pinhole/blob/master/panorama_extraction.ipynb
import sys

import math
import os
import glob
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation network")
    parser.add_argument("--process_interval", help="frame process interval", default=60, type=int)
    parser.add_argument("--create_subfoler", help="create subfoler", default=1, type=int)
    parser.add_argument("--input_video", help="input video path", type=str)
    parser.add_argument(
        "--ignore_prefix",
        help="video filter header, split with ,",
        default="",
        type=str,
    )
    parser.add_argument("--images", help="output folder", default="images", type=str)
    args = parser.parse_args()

    return args


def find_mp4_files(input_dir, ignore_prefix):
    if not os.path.isdir(input_dir):
        raise ValueError(f"invalid path：{input_dir}")

    all_mp4_files = glob.glob(os.path.join(input_dir, "*.MP4"))
    all_mp4_files += glob.glob(os.path.join(input_dir, "*.mp4"))

    if len(ignore_prefix) > 0:
        ignore_header = ignore_prefix.split(",")
        filtered_mp4_files = []
        for file in all_mp4_files:
            valid = True
            for header in ignore_header:
                if os.path.basename(file).startswith(header):
                    valid = False
            if not valid:
                continue
            filtered_mp4_files.append(file)
        return filtered_mp4_files
    return all_mp4_files


def read_video_frames(process_interval, video_path, save_folder, create_subfoler):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("cannot open video", video_path)
        return

    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("  - video size : ", video_width, video_height, frame_length)

    # compute angles to use
    # create a folder to save the video
    video_base_path = os.path.join(save_folder, os.path.basename(video_path)[:-4])
    if create_subfoler:
        os.makedirs(video_base_path, exist_ok=True)
        video_base_path = video_base_path + "/"
    else:
        video_base_path = video_base_path + "_"

    image_count = len(glob.glob(os.path.join(video_base_path, "*.jpg")))
    if not create_subfoler:
        image_count = len(glob.glob(video_base_path + "*.jpg"))
    desired_image_count = int(np.ceil(frame_length / process_interval))
    if image_count == desired_image_count:
        print("  - the video has already been processed")
        return

    frame_count = 0
    progress_bar = tqdm(range(0, frame_length), desc="Loading video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % process_interval != 1:
            continue

        if frame_count == 1:
            progress_bar.update(1)
        else:
            progress_bar.update(process_interval)

        image_path = video_base_path + "{:05d}".format(frame_count) + ".jpg"
        cv2.imwrite(image_path, frame)

    progress_bar.close()
    cap.release()


# python dm/panorama/extract_video_images.py --input_video data/gopro_test
if __name__ == "__main__":
    args = parse_args()
    print(args)

    # find all the video inside the folder
    filtered_mp4_files = find_mp4_files(args.input_video, args.ignore_prefix)

    # create folder to save the images
    save_folder = os.path.join(args.input_video, args.images)
    os.makedirs(save_folder, exist_ok=True)

    if len(filtered_mp4_files) == 0:
        print("No video found")
        os.makedirs(os.path.join(save_folder, "no_ordinary_video"), exist_ok=True)
        exit(0)

    # create a file to tell that ordinary video exist, to let colmap choose opencv camera model
    os.makedirs(os.path.join(save_folder, "use_opencv_model"), exist_ok=True)

    for video in filtered_mp4_files:
        print("process", video)
        read_video_frames(args.process_interval, video, save_folder, args.create_subfoler)
    print("Done")
