import re
import sys
import os
import glob
import argparse
import subprocess
from PIL import Image
import piexif
import cv2
import numpy as np
from tqdm import tqdm

# cp -rfv 2025Nanchansi /mnt/gz01/experiment/mobili/reconstruction/2025Nanchansi
# ls /mnt/gz01/experiment/mobili/reconstruction/

# process video using ffmpeg
# ffmpeg -i input.mp4 -vf fps=1/10 output_%04d.png
def extract_frames(video_path, output_dir, interval=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("cannot open video", video_path)
        return

    video_width  = int(cap.get(3))
    video_height = int(cap.get(4))
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("  - video : ", video_width, video_height, frame_length, fps)
    process_interval = int(fps * interval)

    frame_count = 0
    os.makedirs(output_dir, exist_ok=True)

    # check if the image has been processed
    image_count = len(glob.glob(os.path.join(output_dir, "*.jpg")))
    desired_image_count = int(np.ceil(frame_length / process_interval))
    print(f"  - image_count {image_count}, desired_image_count {desired_image_count}")
    if image_count == desired_image_count:
        print("  - the video has already been processed")
        return []

    progress_bar = tqdm(range(0, frame_length), desc="Loading video")
    images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count%process_interval != 1:
            continue

        if frame_count == 1:
            progress_bar.update(1)
        else :
            progress_bar.update(process_interval)

        image_path = os.path.join(output_dir, "{:05d}".format(frame_count) + ".jpg")
        images.append(image_path)
        cv2.imwrite(image_path, frame)
    return images


def parse_srt(srt_path):
    subtitles = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n')

    for block in lines:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        time_str = lines[1].strip()
        start_str, end_str = time_str.split(' --> ')
        start_time = srt_to_seconds(start_str)
        end_time = srt_to_seconds(end_str)
        content = '\n'.join(lines[2:]).strip()

        subtitles.append({
            'start': start_time,
            'end': end_time,
            'content': content
        })
    return subtitles

def srt_to_seconds(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def find_subtitles(subtitles, timestamp):
    subtitle = None
    if timestamp < subtitles[0]['start']:
        subtitle = subtitles[0]['content']
    if timestamp > subtitles[-1]['end']:
        subtitle = subtitles[-1]['content']
    for sub in subtitles:
        if sub['start'] <= timestamp <= sub['end']:
            subtitle = sub['content']
            break
    assert subtitle is not None
    return subtitle

def extract_lat_lon_alt(input_string):
    lat_pattern = r'latitude: ([\d\.-]+)'
    lon_pattern = r'longitude: ([\d\.-]+)'
    alt_pattern = r'rel_alt: ([\d\.-]+)'
    focal_len_pattern = r'focal_len: ([\d\.-]+)'

    lat_match = re.search(lat_pattern, input_string)
    lon_match = re.search(lon_pattern, input_string)
    alt_match = re.search(alt_pattern, input_string)
    focal_len_match = re.search(focal_len_pattern, input_string)

    latitude = float(lat_match.group(1)) if lat_match else None
    longitude = float(lon_match.group(1)) if lon_match else None
    abs_alt = float(alt_match.group(1)) if alt_match else None
    focal_len = float(focal_len_match.group(1)) if focal_len_match else None

    return latitude, longitude, abs_alt, focal_len

def to_deg(value, ref):
    """
    Converts decimal coordinates into degrees, minutes, and seconds format.
    Returns a tuple: ((degrees, 1), (minutes, 1), (seconds * 100, 100)), ref.
    """
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    seconds = (value - degrees - minutes / 60) * 3600

    return ((degrees, 1), (minutes, 1), (int(seconds * 100), 100)), ref


def add_gps_exif(image_path, output_path, lat, lon, alt, focal_len):
    """
    Adds GPS metadata including altitude to an image's EXIF and saves the output.
    Parameters:
        - image_path: Path to the input image.
        - output_path: Path to save the modified image.
        - lat: Latitude in decimal degrees (positive for N, negative for S).
        - lon: Longitude in decimal degrees (positive for E, negative for W).
        - alt: Altitude in meters (positive above sea level, negative below sea level).
    """
    # Open the image
    img = Image.open(image_path)

    # Get or initialize EXIF data
    # exif_dict = piexif.load(img.info.get("exif", b""))
    try:
        exif_dict = piexif.load(img.info['exif'])
    except KeyError:
        exif_dict = {}
    # exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = focal_len

    # Convert GPS data
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b'N' if lat >= 0 else b'S',
        piexif.GPSIFD.GPSLatitude: to_deg(abs(lat), b'N')[0],
        piexif.GPSIFD.GPSLongitudeRef: b'E' if lon >= 0 else b'W',
        piexif.GPSIFD.GPSLongitude: to_deg(abs(lon), b'E')[0],
        piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1,  # 0 for above sea level, 1 for below
        piexif.GPSIFD.GPSAltitude: (int(abs(alt) * 100), 100),  # Altitude in meters with 2 decimal precision
    }

    # Add GPS data to EXIF
    exif_dict["GPS"] = gps_ifd

    # Save the modified image
    exif_bytes = piexif.dump(exif_dict)
    img.save(output_path, "jpeg", exif=exif_bytes)
    # print(f"GPS data with altitude added to {output_path}")


def process_video(video_path, output_dir, image_interval_sec):
    subtitle_path = video_path[:-3] + "SRT"  # 替换为你的字幕文件路径

    print("===> extract images ", video_path, image_interval_sec)
    images = extract_frames(video_path, output_dir, image_interval_sec);

    if len(images) == 0:
        return

    print("===> extract srt file")
    subtitles = parse_srt(subtitle_path)

    print(f"===> Parsed {len(subtitles)} subtitles.")

    for i in range(len(images)):
        timestamp = i * image_interval_sec
        subtitle = find_subtitles(subtitles, timestamp)
        lat, lon, alt, focal_len = extract_lat_lon_alt(subtitle)
        add_gps_exif(images[i], images[i], lat, lon, alt, focal_len)

    print("==> Done", video_path)


# video_path = "../../ADATA/DJI25/videos0210/DJI_20250210204155_0113_D.MP4" 
# output_dir = "test"
# process_video(video_path, output_dir)
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--image_interval_sec', help='frame process interval', default=1.0, type=float)
    parser.add_argument('--input_video',
                        help='input 360 video path',
                        type=str)
    args = parser.parse_args()
    return args

# python dm/colmap/drone_image_extractor.py --input_video data/DJI_test
if __name__ == "__main__":
    args = parse_args()

    print("===> Process all the video", args.input_video)
    videos = glob.glob(args.input_video + "/DJI*.MP4")
    if len(videos) == 0:
      print("No video found")
      exit(0)

    # create folder to save the images
    save_folder = os.path.join(args.input_video, "images")
    os.makedirs(save_folder, exist_ok=True)

    for video in videos:
        video_name = os.path.basename(video)[:-4]
        output_path = os.path.join(save_folder, video_name)
        process_video(video, output_path, args.image_interval_sec)
    print("Done")
