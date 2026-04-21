import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from pyzbar import pyzbar
from qrcode_detection_yolo import QRCodeDetector


def generate_qr_masks(image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(image_dir.rglob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        decoded_objs = pyzbar.decode(img)

        if len(decoded_objs) > 0:
            for obj in decoded_objs:
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([(p.x, p.y) for p in points], dtype=np.float32))
                    points = hull.reshape(-1, 2)
                else:
                    points = np.array([(p.x, p.y) for p in points], dtype=np.float32)

                points = points.astype(np.int32)

                area = cv2.contourArea(points)
                if area <= 10:
                    print(f"⚠️ Degenerate QR in {img_path.relative_to(image_dir)}, skipped.")
                    continue

                cv2.fillConvexPoly(mask, points, 0)
                print(f"✅ QR detected in {img_path.relative_to(image_dir)}, masked {points.tolist()}")
        else:
            print(f"No QR found in {img_path.relative_to(image_dir)}")

        rel_path = img_path.relative_to(image_dir)
        save_path = mask_dir / rel_path.with_name(rel_path.name + ".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path), mask)

    print(f"\n🎉 All masks saved under: {mask_dir}")


def generate_qr_masks_yolo(image_dir, mask_dir, detect_model_path, trust_threshold):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    yolo_detector = QRCodeDetector(detect_model_path)

    for img_path in sorted(image_dir.rglob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        decoded_objs = yolo_detector.detect_qrcode(img, trust_threshold)

        if len(decoded_objs) > 0:
            for obj in decoded_objs:
                points = obj["polygon"]
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([(p[0], p[1]) for p in points], dtype=np.float32))
                    points = hull.reshape(-1, 2)
                else:
                    points = np.array([(p[0], p[1]) for p in points], dtype=np.float32)

                points = points.astype(np.int32)

                area = cv2.contourArea(points)
                if area <= 10:
                    print(f"⚠️ Degenerate QR in {img_path.relative_to(image_dir)}, skipped.")
                    continue

                cv2.fillConvexPoly(mask, points, 0)
                print(f"✅ QR detected in {img_path.relative_to(image_dir)}, masked {points.tolist()}")
        else:
            print(f"No QR found in {img_path.relative_to(image_dir)}")

        rel_path = img_path.relative_to(image_dir)
        save_path = mask_dir / rel_path.with_name(rel_path.name + ".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path), mask)

    print(f"\n🎉 All masks saved under: {mask_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate QR mask using pyzbar")
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--mask_dir", type=str)
    parser.add_argument(
        "--detect_model_path",
        default="/mnt/data/yeliu/Dev/GaussianSplatting/dm/colmap/model/best-yolov8.pt",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    image_folder = args.image_dir
    mask_folder = args.mask_dir
    detect_model_path = args.detect_model_path
    generate_qr_masks_yolo(image_folder, mask_folder, detect_model_path, 0.75)
