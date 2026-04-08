import os
from datetime import datetime
import argparse
import collections
import piexif
import cv2
import glob
from PIL import Image
from tqdm import tqdm

EXIFTOOL_PATH="exiftool"
GpsMeta = collections.namedtuple(
    "GpsMeta", ["timestamp", "timestampreal", "lat", "lon", "alt"])

def process_video_exif(video_path, output_xml_file):
    # check if output_xml_file exist
    if os.path.isfile(output_xml_file):
        print("  - " + output_xml_file + " exist")
        return True
    try:
        # ./data/Image-ExifTool-13.25/exiftool -ee -G3 -api LargeFileSupport=1 -X -n -b xxx.360 > xxx.xml
        command_line = EXIFTOOL_PATH + " -ee -G3 -api LargeFileSupport=1 -X -n -b " + video_path + " > " + output_xml_file
        print("  - run command line : " + command_line)
        os.system(command_line)
        return True
    except Exception as e:
        print(f"failed when reading exif: {e}")
        return False


def extract_data_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    timestamp = None
    gps_info = {"latitude": None, "longitude": None, "altitude": None}

    gps_data = []
    frames_inlast_second = 0
    current_delta = 0.0

    for line in lines:
        line = line.strip()

        if "<Track4:TimeStamp>" in line and "</Track4:TimeStamp>" in line:
            start = line.find("<Track4:TimeStamp>") + len("<Track4:TimeStamp>")
            end = line.find("</Track4:TimeStamp>")
            timestamp = float(line[start:end].strip())

        if "<Track4:GPSLatitude>" in line:
            start = line.find("<Track4:GPSLatitude>") + len("<Track4:GPSLatitude>")
            end = line.find("</Track4:GPSLatitude>")
            gps_info["latitude"] = float(line[start:end].strip())
        elif "<Track4:GPSLongitude>" in line:
            start = line.find("<Track4:GPSLongitude>") + len("<Track4:GPSLongitude>")
            end = line.find("</Track4:GPSLongitude>")
            gps_info["longitude"] = float(line[start:end].strip())
        elif "<Track4:GPSAltitude>" in line:
            start = line.find("<Track4:GPSAltitude>") + len("<Track4:GPSAltitude>")
            end = line.find("</Track4:GPSAltitude>")
            gps_info["altitude"] = float(line[start:end].strip())

        if all(value is not None for value in gps_info.values()):
            if len(gps_data) > 0 and timestamp == gps_data[-1].timestamp:
                frames_inlast_second += 1
            else:
                if frames_inlast_second > 0:
                    current_delta = 1.0 / frames_inlast_second
                frames_inlast_second = 1

            gps_data.append(GpsMeta(timestamp=timestamp,
                                    timestampreal=timestamp + current_delta * (frames_inlast_second - 1),
                                    lat=gps_info["latitude"], lon=gps_info["longitude"], alt=gps_info["altitude"]))
            gps_info = {"latitude": None, "longitude": None, "altitude": None}
    return gps_data


def to_deg(value, ref):
    """
    Converts decimal coordinates into degrees, minutes, and seconds format.
    Returns a tuple: ((degrees, 1), (minutes, 1), (seconds * 100, 100)), ref.
    """
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    seconds = (value - degrees - minutes / 60) * 3600
    return ((degrees, 1), (minutes, 1), (int(seconds * 100), 100)), ref


def add_gps_exif(image_path, output_path, lat, lon, alt, focus_length):
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

    # https://github.com/hMatoba/Piexif/blob/3422fbe7a12c3ebcc90532d8e1f4e3be32ece80c/piexif/_exif.py#L582
    # https://github.com/colmap/colmap/blob/6be1a35078edd8f391b2c046b367ac977d438a29/src/colmap/sensor/bitmap.cc#L402
    # exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = focal_len
    # exiftool -m -exif:FocalLength=50 00001.jpg
    if focus_length > 0.0:
        if "Exif" not in exif_dict:
            exif_dict["Exif"] = {}
        # exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = (int(focus_length * 1000), 1000)
        exif_dict["Exif"][piexif.ExifIFD.PixelXDimension] = img.size[0]
        exif_dict["Exif"][piexif.ExifIFD.PixelYDimension] = img.size[1]

        exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (int(focus_length * 10), 1)
        exif_dict["Exif"][piexif.ExifIFD.FocalPlaneXResolution] = (1, 1)
        exif_dict["Exif"][piexif.ExifIFD.FocalPlaneResolutionUnit] = 3

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

def get_video_frame_count_and_fps(video_path):
    # get the video information
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("cannot open video", video_path)
        return -1, -1

    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return frame_length, fps


def find_cloest_gps(gps_data, timestamp_raw):
    start_time = gps_data[0].timestampreal
    timestamp = timestamp_raw + start_time
    for gps in gps_data:
        if gps.timestampreal > timestamp:
            return gps
    return gps[-1]


def check_gps_in_exif(image_path):
    try:
        with Image.open(image_path) as img:
            exif_dict = piexif.load(img.info['exif'])
            if "GPS" in exif_dict:
                return True
            return False
    except Exception as e:
        return False


def add_exif_to_image(gps_datas, video_path, focus_length = -1):
    # get the folder with video images
    images_folder = os.path.join(os.path.dirname(video_path), "images")
    images_base_path = os.path.join(images_folder, os.path.basename(video_path)[:-4])
    frame_count, fps = get_video_frame_count_and_fps(video_path)
    if frame_count < 0 or fps < -1:
        print(f"VIDEO ERROR : {video_path} frame_count {frame_count}, fps {fps}")
        exit(1)

    # process for all the images generated
    for image_subfolder in glob.glob(images_base_path + "*"):
        image_paths = glob.glob(image_subfolder + "/*.jpg")
        image_paths.sort()
        if len(image_paths) == 0:
            continue
        print("  - process " + image_subfolder)
        progress_bar = tqdm(range(0, len(image_paths)), desc="Updating Exif")
        for image_path in image_paths:
            if check_gps_in_exif(image_path):
                progress_bar.update(1)
                continue
            image_idx = int(os.path.basename(image_path)[:-4])
            image_timestamp = image_idx * (1.0 / fps)
            gps_data = find_cloest_gps(gps_datas, image_timestamp)
            # print(image_idx, gps_data)
            add_gps_exif(image_path, image_path, gps_data.lat, gps_data.lon, gps_data.alt, focus_length)
            progress_bar.update(1)
        progress_bar.refresh()
        progress_bar.close()

    # prcess for all the images
    image_paths = glob.glob(images_base_path + "*.jpg")
    if len(image_paths) == 0:
        return

    progress_bar = tqdm(range(0, len(image_paths)), desc="Updating Exif")
    for image_path in image_paths:
        # check if GPS exist
        if check_gps_in_exif(image_path):
            progress_bar.update(1)
            continue
        image_idx = int(image_path.split("_")[-1][:-4])
        image_timestamp = image_idx * (1.0 / fps)
        gps_data = find_cloest_gps(gps_datas, image_timestamp)
        # print(image_idx, gps_data)
        add_gps_exif(image_path, image_path, gps_data.lat, gps_data.lon, gps_data.alt)
        progress_bar.update(1)
    progress_bar.refresh()
    progress_bar.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Get gps from exif')
    parser.add_argument('--input_video_folder',
                        help='input video path',
                        type=str)
    args = parser.parse_args()

    return args

# python dm/panorama/gopro_gps_extractor.py --input_video_folder ${MAP_FOLDER}/${SESSION}
if __name__ == "__main__":
    args = parse_args()

    # process for all the videos in the video file
    for input_video in glob.glob(args.input_video_folder + "/*.360"):
        print("  - process", input_video)
        output_xml_file = input_video[:-4] + ".xml"
        exif_ret = process_video_exif(input_video, output_xml_file)
        gps_infos = extract_data_from_file(output_xml_file)
        add_exif_to_image(gps_infos, input_video)

    # add_gps_exif(image_path, output_path, lat, lon, alt)

    print("Done!")
