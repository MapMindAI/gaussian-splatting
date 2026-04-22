# https://github.com/adynathos/panorama_to_pinhole/blob/master/panorama_extraction.ipynb
import sys

import math
import os
import glob
import argparse
import cv2
from panorama_to_pinhole import *
import gopro_gps_extractor
import insta360_meta_extractor
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation network")
    parser.add_argument("--num_divide", help="number of divide", default=4, type=int)
    parser.add_argument("--process_interval", help="frame process interval", default=60, type=int)
    parser.add_argument("--jump_jump", help="jump some of the frames", default=0, type=int)
    parser.add_argument("--input_video", help="input 360 video path", type=str)
    parser.add_argument("--resize_factor", help="resize factor of the output images", default=1.0, type=float)
    args = parser.parse_args()

    return args


def read_video_frames(num_divide, process_interval, jump_jump, resize_factor, video_path, save_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("cannot open video", video_path)
        return -1

    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("  - video size : ", video_width, video_height, frame_length)
    if video_width != video_height * 2:
        print(f"ERROR video, this is not an standard panorama video ({video_width}x{video_height})")
        return -1

    # compute a proper resolution for the output image
    image_width = int(video_width * resize_factor/ num_divide)
    image_height = int(image_width * 0.8)
    output_resolu = (image_width, image_height)
    output_focal = (50 * num_divide) * image_width / 640

    # *focal_length = focal_length_35 / 35.0 * max_size;
    # *focal_length = focal_length_mm / (ccd_width * 10.0) * max_size;
    # focal_length_mm = focal_length * (ccd_width * 10.0) / max_size
    focus_length = output_focal
    print("  - output : ", output_resolu, output_focal, focus_length)

    # compute angles to use
    # create a folder to save the video
    video_base_path = os.path.join(save_folder, os.path.basename(video_path)[:-4])

    already_done = True
    yaw_angles = []
    angle_divide = 360.0 / num_divide
    for i in range(num_divide):
        yaw_angles.append(angle_divide * i)
        os.makedirs(video_base_path + "_" + str(i), exist_ok=True)

        # check number of images in each folder
        image_count = len(glob.glob(os.path.join(video_base_path + "_" + str(i), "*.jpg")))
        desired_image_count = int(np.ceil(frame_length / process_interval))
        if jump_jump:
            if i % 2 == 0:
                desired_image_count = int(desired_image_count / 2)
            else:
                desired_image_count = int(np.ceil(desired_image_count / 2))
        print(f"  - {i} desired_image_count : {desired_image_count}")
        if image_count != desired_image_count:
            already_done = False
    if already_done:
        print("  - the video has already been processed")
        return focus_length

    frame_count = 0
    saved_frame_count = 0
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
        panorama_sampler = panorama_to_sampler(frame)

        for i in range(num_divide):
            if jump_jump and i % 2 != saved_frame_count % 2:
                continue
            image_a = panorama_to_pinhole(panorama_sampler, output_resolu, output_focal, [0, yaw_angles[i], 0])
            image_path = os.path.join(video_base_path + "_" + str(i), "{:05d}".format(frame_count) + ".jpg")
            cv2.imwrite(image_path, image_a)

    progress_bar.close()
    cap.release()
    return focus_length


def process_insta360_videos(input_video_path):
    # insta360 video postfix is insv
    pano_raw_videos = glob.glob(os.path.join(input_video_path, "VID*.insv"))

    for pano_raw_video in pano_raw_videos:
        output_mp4_path = pano_raw_video[:-4] + "mp4"
        output_meta_path = pano_raw_video[:-4] + "meta"

        print(f"process {pano_raw_video} save to {output_mp4_path}")
        try:
            if not os.path.isfile(output_mp4_path):
                command_line = f"""insta360_media_stitcher \
                -inputs {pano_raw_video} -output {output_mp4_path} \
                -stitch_type optflow -enable_stitchfusion \
                -output_size 8000x4000 -bitrate 150000000 \
                -enable_h265_encoder -enable_flowstate -enable_directionlock
                """
                print("  - run command line : " + command_line)
                os.system(command_line)

            if not os.path.isfile(output_meta_path):
                command_line = f"exiftool -ee -m {pano_raw_video} > {output_meta_path}"
                print("  - run command line : " + command_line)
                os.system(command_line)

        except Exception as e:
            print(f"failed when processing insta360_media_stitcher: {e}")


# python mapmind/panorama/extract_pinhole_images.py --input_video data/insta360_test/
if __name__ == "__main__":
    args = parse_args()

    process_insta360_videos(args.input_video)

    # find all the video inside the folder
    # go pro video prefix is GS
    pano_videos = glob.glob(os.path.join(args.input_video, "GS*.MP4"))
    pano_videos += glob.glob(os.path.join(args.input_video, "GS*.mp4"))
    # insta360 video prefix is VID
    pano_videos += glob.glob(os.path.join(args.input_video, "VID*.mp4"))

    if len(pano_videos) == 0:
        print("No pano video found")
        exit(0)

    # create folder to save the images
    save_folder = os.path.join(args.input_video, "images")
    os.makedirs(save_folder, exist_ok=True)

    has_gps = False
    for video_path in pano_videos:
        print("process", video_path)
        focus_length = read_video_frames(
            args.num_divide,
            args.process_interval,
            args.jump_jump,
            args.resize_factor,
            video_path,
            save_folder,
        )
        if focus_length < 0:
            continue

        # add exif from .360 files (of GoPro)
        video_path_360 = video_path[:-4] + ".360"
        output_xml_file = video_path[:-4] + ".xml"
        output_meta_file = video_path[:-4] + ".meta"
        exif_ret = gopro_gps_extractor.process_video_exif_gopro(video_path_360, output_xml_file)
        if exif_ret:
            has_gps = True
            gps_infos = gopro_gps_extractor.extract_data_from_file(output_xml_file)
            gopro_gps_extractor.add_exif_to_image(gps_infos, video_path_360, focus_length)
        elif os.path.isfile(output_meta_file):
            # this is insta360 file
            gps_list = insta360_meta_extractor.read_gps_from_meta(output_meta_file)
            has_gps = len(gps_list) > 0
            if has_gps:
                insta360_meta_extractor.add_exif_to_image(gps_list, video_path, focus_length)

    # create a file to tell that image gps exist
    if has_gps:
        print("GPS data obtained!")
        os.makedirs(os.path.join(save_folder, "image_with_gps"), exist_ok=True)

    print("Done!")
