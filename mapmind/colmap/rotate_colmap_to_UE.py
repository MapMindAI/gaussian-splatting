import os
import cv2
import numpy as np
import argparse
from qrcode.transform_colmap_model_qrcode import apply_sim3_to_colmap_model


def rotate_colmap_to_UE(colmap_model_path, input_sparse_path="sparse/0/", output_sparse_path=None):
    # x->x, y->z, z->-y
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    t = np.array([0, 0, 0])
    s = 1.0

    apply_sim3_to_colmap_model(colmap_model_path, input_sparse_path, R, t, s, output_sparse_path)
    print(f"Rotated model saved to {output_sparse_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--colmap_model_path", type=str, required=True, help="path to colmap model")
    p.add_argument(
        "--input_sparse_path",
        default="sparse/0/",
        help="input sparse model path under colmap_model_dir",
    )
    p.add_argument(
        "--output_sparse_path",
        default="sparse_test/0/",
        help="if set, write scaled model to this path instead of overwriting",
    )
    args = p.parse_args()

    rotate_colmap_to_UE(args.colmap_model_path, args.input_sparse_path, args.output_sparse_path)
