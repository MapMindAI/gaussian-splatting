import cv2
from ultralytics import YOLO
import torch
import numpy as np


class QRCodeDetector:
    def __init__(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"Load yolo model from {model_path}")

    def detect_qrcode(self, img, trust_threshold):
        predictions = self.model(img)

        result = []
        for det in predictions:
            if det.boxes is not None and len(det.boxes) > 0:
                for box in det.boxes:
                    if box.conf < trust_threshold:
                        continue

                    obj = {}
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    points = [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                    ]

                    obj["polygon"] = points
                    result.append(obj)

        return result


def draw_img(img, predictions, outfile):
    for obj in predictions:
        points = np.array(obj["polygon"]).astype(np.int32)
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imwrite(outfile, img)


if __name__ == "__main__":
    model_path = "/mnt/data/yeliu/Dev/GaussianSplatting/dm/colmap/model/best-yolov8.pt"
    detector = QRCodeDetector(model_path)

    image_folder = "/mnt/data/yeliu/gaussian_splatting/GoPro/qrcode_test/room1_sequence/images/"
    image_name = "/GS010101_2/00911.jpg"
    image_file = image_folder + image_name
    img = cv2.imread(image_file)

    predictions = None
    if img is not None:
        predictions = detector.detect_qrcode(img, 0.7)

    outfile = "/mnt/data/yeliu/Dev/GaussianSplatting/dm/colmap/model/detect.jpg"
    if predictions is not None:
        draw_img(img, predictions, outfile)
        print(f"detect {len(predictions)} QRcodes!")
