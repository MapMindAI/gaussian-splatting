import os
from datetime import datetime, timedelta
import argparse
import collections
import cv2
import glob
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import re
from gopro_gps_extractor import add_gps_exif, GpsMeta, add_exif_to_image

ImuMeta = collections.namedtuple(
    "ImuMeta", ["timestamp", "acc", "gyr"])


def process_video_exif_insta360(video_path, output_meta_file):
    if not os.path.isfile(video_path):
        print("  - " + video_path + " not exist")
        return False
    # check if output_xml_file exist
    if os.path.isfile(output_meta_file):
        print("  - " + output_meta_file + " exist")
        return True
    try:
        command_line = f"exiftool -ee -m {video_path} > {output_meta_file}"
        print("  - run command line : " + command_line)
        os.system(command_line)
        return True
    except Exception as e:
        print(f"failed when reading exif: {e}")
        return False


def read_imu_from_meta(meta_file_path):
    records = []

    acc = None
    gyr = None
    timestamp = None

    with open(meta_file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            key, value = [x.strip() for x in line.split(":", 1)]

            if key == "Accelerometer":
                acc = [float(x) for x in value.split()]
            elif key == "Angular Velocity":
                gyr = [float(x) for x in value.split()]
            elif key == "Time Code":
                timestamp = float(value)

                # finish one sample only when all parts exist
                if acc is not None and gyr is not None:
                    records.append(
                        ImuMeta(
                            timestamp=timestamp,
                            acc=acc,
                            gyr=gyr,
                        )
                    )
                    acc = None
                    gyr = None
                    timestamp = None
    return records


def dms_to_decimal(text):
    """
    Convert:
        22 deg 46' 54.52" N
        113 deg 30' 50.35" E
    to decimal degrees.
    """
    m = re.match(
        r'^\s*(\d+)\s+deg\s+(\d+)\'\s+([\d.]+)"\s+([NSEW])\s*$',
        text
    )
    if not m:
        raise ValueError(f"Invalid DMS format: {text}")

    deg = float(m.group(1))
    minute = float(m.group(2))
    second = float(m.group(3))
    direction = m.group(4)

    value = deg + minute / 60.0 + second / 3600.0
    if direction in ("S", "W"):
        value = -value
    return value


def parse_gps_datetime(text):
    """
    Convert:
        2026:04:17 06:21:44Z
    to datetime.
    """
    try:
        return datetime.strptime(text, "%Y:%m:%d %H:%M:%SZ")
    except Exception as e:
        # print(f"failed when decode time {text}: {e}")
        return None


def parse_create_datetime(text):
    """
    Convert:
        2026:04:17 06:19:37
    to datetime.
    """
    return datetime.strptime(text, "%Y:%m:%d %H:%M:%S")


def parse_altitude(text):
    """
    Convert:
        3.9368 m
    to float meters.
    """
    return float(text.replace("m", "").strip())


def read_gps_from_meta(path):
    records = []

    create_date = None
    speed = None
    track = None
    altitude = None
    base_timestamp = None
    latitude = None
    longitude = None

    last_base_timestamp = None
    duplicate_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            key, value = [x.strip() for x in line.split(":", 1)]

            if key == "Create Date":
                create_date = parse_create_datetime(value)

            if key == "GPS Speed":
                speed = float(value)

            elif key == "GPS Track":
                track = float(value)

            elif key == "GPS Altitude":
                altitude = parse_altitude(value)

            elif key == "GPS Date/Time":
                base_timestamp = parse_gps_datetime(value)

            elif key == "GPS Latitude":
                latitude = dms_to_decimal(value)

            elif key == "GPS Longitude":
                longitude = dms_to_decimal(value)

                # assume longitude is the last field of one GPS block
                if None not in (create_date, speed, track, altitude, base_timestamp, latitude, longitude):
                    # GPS timestamps are stored with only 1-second precision, while data arrives at 10 Hz.
                    # If consecutive records have the same GPS Date/Time, add 0.1 s per repeated sample
                    # so each record gets a unique timestamp in chronological order.
                    if last_base_timestamp is None or base_timestamp != last_base_timestamp:
                        duplicate_count = 0
                        last_base_timestamp = base_timestamp
                    else:
                        duplicate_count += 1

                    timestamp = base_timestamp + timedelta(seconds=0.1 * duplicate_count)
                    time_since_create = (timestamp - create_date).total_seconds()

                    records.append(
                        GpsMeta(
                            timestamp=timestamp,
                            timestampreal=time_since_create,
                            alt=altitude,
                            lat=latitude,
                            lon=longitude,
                        )
                    )

                    speed = None
                    track = None
                    altitude = None
                    base_timestamp = None
                    latitude = None
                    longitude = None

    return records


def parse_args():
    parser = argparse.ArgumentParser(description='Get gps from exif')
    parser.add_argument('--input_video_folder',
                        help='input video path',
                        type=str)
    args = parser.parse_args()

    return args

# python mapmind/panorama/insta360_meta_extractor.py --input_video_folder data/insta360_test/
if __name__ == "__main__":
    args = parse_args()

    # process for all the videos in the video file
    for input_video in glob.glob(args.input_video_folder + "/*.insv"):
        print("  - process", input_video)
        output_meta_file = input_video[:-4] + "meta"
        exif_ret = process_video_exif_insta360(input_video, output_meta_file)
        imu_list = read_imu_from_meta(output_meta_file)
        gps_list = read_gps_from_meta(output_meta_file)

        add_exif_to_image(gps_list, input_video)

    print("Done!")
