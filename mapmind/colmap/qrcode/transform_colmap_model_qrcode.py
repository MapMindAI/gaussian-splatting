import os
import cv2
import math
import numpy as np
import json
import argparse
from tqdm import tqdm
from scipy.optimize import least_squares

import sys

sys.path.append("scene")
from colmap_loader import *

try:
    from pyzbar import pyzbar

    HAS_PYZBAR = True
except Exception:
    HAS_PYZBAR = False
    print("⚠ pyzbar not available, will use OpenCV QRCodeDetector fallback (may be less robust).")


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix

    Args:
      quat: Quaternion [w, x, y, z]

    Returns:
      3x3 rotation matrix
    """
    w, x, y, z = quat

    # Normalize quaternion
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm > 0:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Convert to rotation matrix
    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )

    return R


def rotation_matrix_to_quaternion(R):
    R = np.array(R)

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        # Case 1: trace > 0
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        # Case 2: R[0,0] is largest
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        # Case 3: R[1,1] is largest
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        # Case 4: R[2,2] is largest
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return np.array([qw, qx, qy, qz])


def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def parse_cameras_txt(path):
    cams = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            # Support common models: PINHOLE (fx fy cx cy), SIMPLE_PINHOLE(f cx cy),
            # SIMPLE_RADIAL(f cx cy k), RADIAL(k1 k2)
            if model.upper() == "PINHOLE":
                fx, fy, cx, cy = params[:4]
                dist = np.zeros(5, dtype=float)
            elif model.upper() == "SIMPLE_PINHOLE":
                f, cx, cy = params[:3]
                fx = fy = f
                dist = np.zeros(5, dtype=float)
            elif model.upper() == "SIMPLE_RADIAL":
                f, cx, cy, k = params[:4]
                fx = fy = f
                dist = np.array([k, 0, 0, 0, 0], dtype=float)
            elif model.upper() in ("RADIAL", "RADIAL6"):
                # RADIAL: f cx cy k1 k2
                f, cx, cy = params[0:3]
                k = params[3:]
                fx = fy = f
                dist = np.zeros(5)
                for i in range(min(len(k), 5)):
                    dist[i] = k[i]
            else:
                # fallback: try to find fx,fy,cx,cy in params
                if len(params) >= 4:
                    fx, fy, cx, cy = params[0:4]
                    dist = np.zeros(5)
                else:
                    raise RuntimeError(f"Unsupported camera model {model} in cameras.txt")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            cams[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "K": K,
                "dist": dist,
            }
    return cams


def parse_points3d_txt(path):
    pts3d = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            error = float(parts[7])
            track_elems = parts[8:]
            track = []
            for i in range(0, len(track_elems), 2):
                img_id = int(track_elems[i])
                kp_idx = int(track_elems[i + 1])
                track.append((img_id, kp_idx))
            pts3d[point_id] = {
                "xyz": np.array([x, y, z], dtype=float),
                "rgb": (r, g, b),
                "error": error,
                "track": track,
            }
    return pts3d


def parse_images_txt(path):
    imgs = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        line = lines[i]
        parts = line.split()
        # format: IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID NAME
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        # next line is 2D points, ignore for now
        imgs[name] = {
            "qw": qw,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "cam_id": cam_id,
            "image_id": image_id,
        }
        i += 2
    return imgs


# ---------- helper ----------
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack binary file with little endian (default)."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def parse_cameras_bin(path):
    cams = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = read_next_bytes(fid, 4, "i")[0]
            model_id = read_next_bytes(fid, 4, "i")[0]
            model_name = {
                0: "SIMPLE_PINHOLE",
                1: "PINHOLE",
                2: "SIMPLE_RADIAL",
                3: "RADIAL",
                4: "OPENCV",
                5: "OPENCV_FISHEYE",
                6: "FULL_OPENCV",
                7: "FOV",
                8: "SIMPLE_RADIAL_FISHEYE",
                9: "RADIAL_FISHEYE",
                10: "THIN_PRISM_FISHEYE",
            }[model_id]
            width = read_next_bytes(fid, 8, "Q")[0]
            height = read_next_bytes(fid, 8, "Q")[0]
            num_params = {
                "SIMPLE_PINHOLE": 3,
                "PINHOLE": 4,
                "SIMPLE_RADIAL": 4,
                "RADIAL": 5,
                "OPENCV": 8,
                "OPENCV_FISHEYE": 4,
                "FULL_OPENCV": 12,
                "FOV": 5,
                "SIMPLE_RADIAL_FISHEYE": 4,
                "RADIAL_FISHEYE": 5,
                "THIN_PRISM_FISHEYE": 12,
            }[model_name]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)

            # build intrinsics & distortion like txt parser
            if model_name == "PINHOLE":
                fx, fy, cx, cy = params[:4]
                dist = np.zeros(5)
            elif model_name == "SIMPLE_PINHOLE":
                f, cx, cy = params[:3]
                fx = fy = f
                dist = np.zeros(5)
            elif model_name == "SIMPLE_RADIAL":
                f, cx, cy, k = params[:4]
                fx = fy = f
                dist = np.array([k, 0, 0, 0, 0])
            elif model_name in ("RADIAL",):
                f, cx, cy, k1, k2 = params[:5]
                fx = fy = f
                dist = np.array([k1, k2, 0, 0, 0])
            else:
                # fallback
                fx, fy, cx, cy = params[:4]
                dist = np.zeros(5)

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            cams[camera_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "K": K,
                "dist": dist,
            }
    return cams


def parse_images_bin(path):
    imgs = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qw, qx, qy, qz = read_next_bytes(fid, 8 * 4, "dddd")
            tx, ty, tz = read_next_bytes(fid, 8 * 3, "ddd")
            cam_id = read_next_bytes(fid, 4, "i")[0]
            name_chars = []
            while True:
                c = read_next_bytes(fid, 1, "c")[0]
                if c == b"\x00":
                    break
                name_chars.append(c.decode("utf-8"))
            name = "".join(name_chars)

            # skip 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2D * (8 * 2 + 8))  # x,y,point3D_id

            imgs[name] = {
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "cam_id": cam_id,
                "image_id": image_id,
            }
    return imgs


def parse_points3d_bin(path):
    pts3d = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point_id = read_next_bytes(fid, 8, "Q")[0]
            x, y, z, r, g, b, error = read_next_bytes(fid, 8 * 6 + 8, "dddBBBd")
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track = []
            for _ in range(track_length):
                image_id, point2D_idx = read_next_bytes(fid, 8, "ii")
                track.append((image_id, point2D_idx))
            pts3d[point_id] = {
                "xyz": np.array([x, y, z], dtype=float),
                "rgb": (r, g, b),
                "error": error,
                "track": track,
            }
    return pts3d


def detect_qr_pyzbar(img):
    dec = pyzbar.decode(img)
    res = []
    for d in dec:
        data = d.data.decode("utf-8")
        # polygon: list of Point objects
        poly = [(p.x, p.y) for p in d.polygon]
        # if polygon has >4 points, approximate convex hull
        if len(poly) > 4:
            poly = cv2.convexHull(np.array(poly, dtype=np.float32)).reshape(-1, 2).tolist()
        res.append({"id": data, "corners": poly})
    return res


def detect_qr_opencv(img):
    # returns list of dicts like pyzbar
    detector = cv2.QRCodeDetector()
    ok, decoded_infos, points, _ = detector.detectAndDecodeMulti(img)
    res = []
    print(f"ok: {ok}")
    if ok and points is not None:
        for idx, pts in enumerate(points):
            data = decoded_infos[idx] if decoded_infos is not None else ""
            poly = pts.reshape(-1, 2).tolist()
            res.append({"id": data, "corners": poly})
    return res


def detect_qrcodes(img):
    if HAS_PYZBAR:
        return detect_qr_pyzbar(img)
    else:
        return detect_qr_opencv(img)


def order_corners_clockwise(pts):
    arr = np.array(pts, dtype=float)
    if arr.shape[0] != 4:
        # 如果检测出来 >4（如 pyzbar polygon 包含额外点），先取凸包并挑 4 个顶点
        hull = cv2.convexHull(arr.astype(np.float32)).reshape(-1, 2)
        if hull.shape[0] >= 4:
            # 如果 hull>4，选最远的 4 个角点（通常不常见）
            arr = hull
        else:
            # fallback: use original
            pass

    c = arr.mean(axis=0)
    vecs = arr - c
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])  # -pi..pi
    idx = np.argsort(angles)  # counterclockwise
    arr = arr[idx]

    # ensure we have tl,tr,br,bl in order (clockwise starting at top-left)
    # after sorting counterclockwise, find which one is top-left by y then x
    # build simple mapping: compute bounding box center and use signs
    # compute quadrant signs relative to center
    ordered = [None, None, None, None]  # tl, tr, br, bl
    for p in arr:
        if p[0] < c[0] and p[1] < c[1]:
            ordered[0] = p  # tl
        elif p[0] >= c[0] and p[1] < c[1]:
            ordered[1] = p  # tr
        elif p[0] >= c[0] and p[1] >= c[1]:
            ordered[2] = p  # br
        else:
            ordered[3] = p  # bl

    # if any is None (degenerate), fallback to angle-based assignment
    if any(x is None for x in ordered):
        # assume arr currently CCW starting at some corner
        # produce clockwise starting from top-left by rotating and flipping as needed
        # find candidate with smallest y (top), if tie choose smallest x -> top-left
        tl_idx = np.argmin(arr[:, 1] + arr[:, 0] * 0.001)
        # reorder so tl_idx is first, then clockwise
        arr_clockwise = np.roll(arr[::-1], -tl_idx, axis=0)  # reverse->clockwise
        ordered = [
            arr_clockwise[0],
            arr_clockwise[1],
            arr_clockwise[2],
            arr_clockwise[3],
        ]

    return [tuple(map(float, p)) for p in ordered]


def triangulate_qr_corners(records, cams, imgs, qr_id):
    points3d = []
    for corner_idx in range(4):
        A = []
        for rec in records:
            cam = cams[imgs[rec["image"]]["cam_id"]]
            K = np.asarray(cam["K"], dtype=np.float64)
            quat = np.array(
                [
                    imgs[rec["image"]]["qw"],
                    imgs[rec["image"]]["qx"],
                    imgs[rec["image"]]["qy"],
                    imgs[rec["image"]]["qz"],
                ],
                dtype=np.float64,
            )
            R_cw = quaternion_to_rotation_matrix(quat)
            t_cw = np.array(
                [
                    imgs[rec["image"]]["tx"],
                    imgs[rec["image"]]["ty"],
                    imgs[rec["image"]]["tz"],
                ],
                dtype=np.float64,
            ).reshape(3, 1)
            P = K @ np.hstack([R_cw, t_cw])

            u, v = rec["corners"][corner_idx]
            # u*P[2,:]-P[0,:], v*P[2,:]-P[1,:]
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])

            img_name = rec["image"]
            if corner_idx == 0:  # 只打印一次
                print(f"id: {qr_id}, img: {img_name}, t_cw: {t_cw.flatten()}, t_wc: {(-R_cw.T @ t_cw).flatten()}")

        A = np.asarray(A)
        if A.shape[0] < 4:
            points3d.append(None)
            continue

        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        X = X_h[:3] / X_h[3]
        points3d.append(X)

    return points3d


def get_qr_center(points3d):
    pts = [p for p in points3d if p is not None]
    if len(pts) < 4:
        return None
    return np.mean(pts, axis=0)


def _init_qr_centers_by_triangulation(all_observations, cams, imgs):
    grouped = {}
    for obs in all_observations:
        grouped.setdefault(obs["id"], []).append({"image": obs["image"], "corners": obs["corners"]})

    init_centers = {}
    init_obj_corners = {}
    for qr_id, records in grouped.items():
        if len(records) < 2:
            continue
        pts3d = triangulate_qr_corners(records, cams, imgs, qr_id)
        if any(p is None for p in pts3d):
            continue

        # center
        c = get_qr_center(pts3d)
        if c is not None:
            init_centers[qr_id] = c.astype(float)

        # local corners
        pts3d_centered = np.array(pts3d) - c.reshape(1, 3)
        init_obj_corners[qr_id] = pts3d_centered.astype(float)

    return init_centers, init_obj_corners


def estimate_qrcode_poses_global(all_observations, cams, imgs, verbose=2):
    """
    all_observations: list of dict(
        id=qr_id, image=img_name, corners=(4,2) numpy array(float)
    )
    cams: {cam_id: {"K":, "dist":}}
    imgs: {image_name: {"cam_id":, "qw":,"qx":,"qy":,"qz":,"tx":,"ty":,"tz":}}
    """
    qr_ids = sorted(set(obs["id"] for obs in all_observations))

    default_half = 0.5
    default_obj_corners = np.array(
        [
            [-default_half, default_half, 0.0],
            [default_half, default_half, 0.0],
            [default_half, -default_half, 0.0],
            [-default_half, -default_half, 0.0],
        ],
        dtype=float,
    )

    init_centers, init_obj_corners = _init_qr_centers_by_triangulation(all_observations, cams, imgs)
    param0 = []
    for qr_id in qr_ids:
        rvec0 = np.zeros(3, dtype=float)
        if qr_id in init_centers:
            t0 = init_centers[qr_id]
        else:
            t0 = np.array([0.0, 0.0, 1.0], dtype=float)
        param0.append(np.hstack([rvec0, t0]))
    param0 = np.hstack(param0)

    view_cache = {}
    for img_name, info in imgs.items():
        folder_name = img_name.split("/")[0]
        folder_num = int(folder_name.split("_")[1])

        cam = cams[info["cam_id"]]
        K = cam["K"].astype(float)
        dist = cam["dist"].astype(float) if cam["dist"] is not None else None
        quat = np.array([info["qw"], info["qx"], info["qy"], info["qz"]], dtype=float)

        R_cw = quaternion_to_rotation_matrix(quat)
        t_cw = np.array([info["tx"], info["ty"], info["tz"]], dtype=float).reshape(3, 1)
        view_cache[img_name] = (K, dist, R_cw, t_cw)

    def residuals(p):
        res = []
        qr_params = {qr_id: p[i * 6 : (i + 1) * 6] for i, qr_id in enumerate(qr_ids)}

        for obs in all_observations:
            qr_id = obs["id"]
            corners_2d = obs["corners"].astype(float)  # (4,2)
            K, dist, R_cw, t_cw = view_cache[obs["image"]]

            rvec = qr_params[qr_id][:3]
            t_w = qr_params[qr_id][3:].reshape(3, 1)
            R_wq = rodrigues_to_R(rvec)

            obj_corners_qr = init_obj_corners.get(qr_id, default_obj_corners)
            Xw = (R_wq @ obj_corners_qr.T + t_w).T  # (4,3)

            Xc = (R_cw @ Xw.T + t_cw).T  # (4,3)

            imgpts, _ = cv2.projectPoints(
                Xc.astype(np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                K,
                dist,
            )
            proj = imgpts.reshape(-1, 2)

            res.append((proj - corners_2d).ravel())

        return np.concatenate([np.atleast_1d(r) for r in res])

    result = least_squares(
        residuals,
        param0,
        method="trf",
        loss="huber",
        f_scale=3.0,
        verbose=verbose,
        max_nfev=200,
    )

    qr_poses = {"anchor_num": len(qr_ids), "anchors": {}}
    for i, qr_id in enumerate(qr_ids):
        rvec = result.x[i * 6 : (i * 6 + 3)]
        t = result.x[i * 6 + 3 : (i + 1) * 6]
        R_wq = rodrigues_to_R(rvec)
        qr_poses["anchors"][qr_id] = dict(R=R_wq, t=t)

    edge_len_stats = {}
    for qr_id, pose in qr_poses["anchors"].items():
        R_wq = pose["R"]
        t_wq = np.array(pose["t"], dtype=float).reshape(3, 1)
        obj_corners_qr = init_obj_corners.get(qr_id, default_obj_corners)
        Xw = (R_wq @ obj_corners_qr.T + t_wq).T  # (4,3)
        e01 = np.linalg.norm(Xw[0] - Xw[1])
        e12 = np.linalg.norm(Xw[1] - Xw[2])
        e23 = np.linalg.norm(Xw[2] - Xw[3])
        e30 = np.linalg.norm(Xw[3] - Xw[0])
        edges = [e01, e12, e23, e30]
        edge_len_stats[qr_id] = {"edge_lens": edges, "mean_edge": float(np.mean(edges))}

    pairwise_dists_est = {}
    ids = list(qr_poses["anchors"].keys())
    centers = {qid: np.array(qr_poses["anchors"][qid]["t"], dtype=float) for qid in ids}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            d = float(np.linalg.norm(centers[id1] - centers[id2]))
            pairwise_dists_est[(id1, id2)] = d

    return qr_poses, edge_len_stats, pairwise_dists_est, init_obj_corners


def estimate_scale_from_qr(edge_len_stats, pairwise_dists_est, qr_size_m=None, qr_distances_m=None):
    """
    return：
      {
        'scale_candidates': {'from_size': [...], 'from_pairs': [...]},
        'scale_final': float or None,
        'used': {'size': bool, 'pairs': bool}
      }
    """
    scales_from_size = []
    if qr_size_m is not None:
        means = [info["mean_edge"] for info in edge_len_stats.values() if info["mean_edge"] > 0]
        if len(means) > 0:
            scales_from_size = [qr_size_m / m for m in means]

    scales_from_pairs = []
    if qr_distances_m:
        for (id1, id2), d_true in qr_distances_m.items():
            key = (id1, id2) if (id1, id2) in pairwise_dists_est else (id2, id1)
            if key in pairwise_dists_est:
                d_est = pairwise_dists_est[key]
                if d_est > 0:
                    scales_from_pairs.append(d_true / d_est)

    all_candidates = []
    if len(scales_from_size) > 0:
        all_candidates += scales_from_size
    if len(scales_from_pairs) > 0:
        all_candidates += scales_from_pairs

    scale_final = float(np.median(all_candidates)) if len(all_candidates) > 0 else None

    return {
        "scale_candidates": {
            "from_size": scales_from_size,
            "from_pairs": scales_from_pairs,
        },
        "scale_final": scale_final,
        "used": {
            "size": len(scales_from_size) > 0,
            "pairs": len(scales_from_pairs) > 0,
        },
    }


def estimate_qrcode_poses_optimize(
    colmap_model_dir,
    input_sparse_path,
    image_dir,
    qr_size_m=None,
    qr_distances_m=None,
    output_dir=None,
):
    sparse_dir = os.path.join(colmap_model_dir, input_sparse_path)
    try:
        cams = parse_cameras_txt(os.path.join(sparse_dir, "cameras.txt"))
        imgs = parse_images_txt(os.path.join(sparse_dir, "images.txt"))
    except Exception:
        cams = parse_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))
        imgs = parse_images_bin(os.path.join(sparse_dir, "images.bin"))

    image_dir = os.path.join(colmap_model_dir, image_dir)

    # Get all detections for every qrcode IDs
    all_observations = []
    for img_name, info in tqdm(imgs.items(), desc="Processing images"):
        # Read image
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"!! failed to load {img_path}")
            continue

        # Detect QR codes and fill detections_per_id
        detections = detect_qrcodes(img)
        if not detections:
            continue
        print(f"Detect {len(detections)} qrcode in image: {img_name}")

        for det in detections:
            qr_id = det["id"] if det["id"] else "unknown"
            pts2d = det["corners"]
            if len(pts2d) < 4:
                continue
            # order corners to [tl,tr,br,bl]
            ordered = order_corners_clockwise(pts2d)

            obs = {
                "id": qr_id,
                "image": img_name,
                "corners": np.array(ordered, dtype=float),
            }
            all_observations.append(obs)

    # Global optimize
    (
        qr_poses,
        edge_len_stats,
        pairwise_dists_est,
        init_obj_corners,
    ) = estimate_qrcode_poses_global(all_observations, cams, imgs, verbose=2)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, "qr_visualizations", "fuse_pose")
        draw_projection(
            all_observations,
            qr_poses,
            cams,
            imgs,
            image_dir,
            init_obj_corners=init_obj_corners,
            save_dir=vis_dir,
        )

    scale_info = estimate_scale_from_qr(
        edge_len_stats=edge_len_stats,
        pairwise_dists_est=pairwise_dists_est,
        qr_size_m=qr_size_m,
        qr_distances_m=qr_distances_m,
    )

    return {
        "qr_poses": qr_poses,
        "edge_len_stats": edge_len_stats,
        "pairwise_dists_est": pairwise_dists_est,
        "scale_info": scale_info,
    }


def draw_projection(all_observations, qr_poses, cams, imgs, image_dir, init_obj_corners, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    anchors = {}
    for obs in all_observations:
        anchors.setdefault(obs["id"], []).append({"image": obs["image"], "corners": obs["corners"]})

    default_half = 0.5
    default_obj_corners = np.array(
        [
            [-default_half, default_half, 0.0],
            [default_half, default_half, 0.0],
            [default_half, -default_half, 0.0],
            [-default_half, -default_half, 0.0],
        ],
        dtype=float,
    )

    for qr_id, records in anchors.items():
        if qr_id not in qr_poses["anchors"]:
            continue
        pose = qr_poses["anchors"][qr_id]
        R_wq = pose["R"]
        t_wq = np.array(pose["t"], dtype=float).reshape(3, 1)

        obj_corners_qr = init_obj_corners.get(qr_id, default_obj_corners)
        Xw = (R_wq @ obj_corners_qr.T + t_wq).T  # (4,3)

        print("--------------------------")
        print(f"QR ID: {qr_id}\n R_wq:\n{R_wq}, t_wq: {t_wq.flatten()}")
        print(f"Xw: {Xw}\n")

        merge_list = []
        for rec in records[:8]:
            img_name = rec["image"]
            info = imgs[img_name]
            cam = cams[info["cam_id"]]
            K = cam["K"].astype(float)
            dist = cam["dist"].astype(float) if cam["dist"] is not None else None

            quat = np.array([info["qw"], info["qx"], info["qy"], info["qz"]], dtype=float)
            R_cw = quaternion_to_rotation_matrix(quat)
            t_cw = np.array([info["tx"], info["ty"], info["tz"]], dtype=float).reshape(3, 1)

            Xc = (R_cw @ Xw.T + t_cw).T

            print(f"R_cw:\n{R_cw}, t_cw: {t_cw.flatten()}")
            print(f"Xc: {Xc}")

            imgpts, _ = cv2.projectPoints(
                Xc.astype(np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                K,
                dist,
            )
            imgpts = np.int32(imgpts.reshape(-1, 2))

            img = cv2.imread(os.path.join(image_dir, img_name))
            if img is None:
                continue

            for i in range(4):
                cv2.line(img, tuple(imgpts[i]), tuple(imgpts[(i + 1) % 4]), (0, 0, 255), 2)
                cv2.circle(img, tuple(imgpts[i]), 4, (0, 255, 0), -1)
                cv2.putText(
                    img,
                    str(i),
                    (imgpts[i][0] - 8, imgpts[i][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            det = np.int32(rec["corners"])
            for i in range(4):
                cv2.circle(img, tuple(det[i]), 3, (255, 255, 0), -1)

            merge_list.append(img)

        if merge_list:
            merged = cv2.hconcat(merge_list)
            save_path = os.path.join(save_dir, f"{qr_id}_merged.jpg")
            cv2.imwrite(save_path, merged)
            print(f"✅ Saved merged image: {save_path}")


def apply_sim3_to_colmap_model(
    colmap_model_dir,
    input_sparse_path,
    R,
    t,
    scale,
    output_sparse_path=None,
    qr_poses=None,
):
    sparse_dir = os.path.join(colmap_model_dir, input_sparse_path)
    imgs, cams, pts3d = read_colmap_model(sparse_dir)

    print(f"Applying Sim(3) to sparse model (scale={scale:.6f}) ...")
    print(f"Loaded {len(cams)} cameras, {len(imgs)} images, {len(pts3d)} points3D.")

    new_images = {}
    for img_id, info in imgs.items():
        R_cw = quaternion_to_rotation_matrix(info.qvec)
        t_cw = info.tvec.reshape(3, 1)

        C_w = -R_cw.T @ t_cw  # (3,1)

        C_w_new = scale * (R @ C_w) + t.reshape(3, 1)
        R_cw_new = R_cw @ R.T
        t_cw_new = -R_cw_new @ C_w_new

        qvec_new = rotation_matrix_to_quaternion(R_cw_new)
        tvec_new = t_cw_new.flatten()
        new_images[img_id] = Image(
            id=info.id,
            qvec=qvec_new,
            tvec=tvec_new,
            camera_id=info.camera_id,
            name=info.name,
            xys=info.xys,
            point3D_ids=info.point3D_ids,
        )

    new_pts3d = {}
    for pt_id, info in pts3d.items():
        xyz_new = scale * (R @ info.xyz) + t
        new_pts3d[pt_id] = Point3D(
            id=info.id,
            xyz=xyz_new.reshape(3),
            rgb=info.rgb,
            error=info.error,
            image_ids=info.image_ids,
            point2D_idxs=info.point2D_idxs,
        )

    if qr_poses is not None:
        new_qr_poses = {}
        for qr_id, pose in qr_poses.items():
            R_qr = np.array(pose["R"], dtype=float)
            t_qr = np.array(pose["t"], dtype=float)

            R_new = R @ R_qr
            t_new = scale * R @ t_qr + t

            new_qr_poses[qr_id] = dict(R=R_new.tolist(), t=t_new.tolist())

        qr_poses_json_path = os.path.join(colmap_model_dir, "qr_poses.json")
        with open(qr_poses_json_path, "w", encoding="utf-8") as f:
            json.dump(new_qr_poses, f, indent=2, ensure_ascii=False)

    if output_sparse_path:
        output_model_path = os.path.join(colmap_model_dir, output_sparse_path)
        os.makedirs(output_model_path, exist_ok=True)
    else:
        output_model_path = sparse_dir

    write_images_text(new_images, os.path.join(output_model_path, "images.txt"))
    write_cameras_text(cams, os.path.join(output_model_path, "cameras.txt"))
    write_points3D_text(new_pts3d, os.path.join(output_model_path, "points3D.txt"))
    write_images_binary(new_images, os.path.join(output_model_path, "images.bin"))
    write_cameras_binary(cams, os.path.join(output_model_path, "cameras.bin"))
    write_points3D_binary(new_pts3d, os.path.join(output_model_path, "points3D.bin"))

    print(f"Output done! Wrote to {output_model_path}")


def recover_scale(colmap_model_dir, input_sparse_path, qr_poses, scale, output_sparse_path=None):
    if (scale is None) or (scale <= 0.1) or (scale > 1e4):
        print("!! Invalid scale, skip recovering.")
        return

    apply_sim3_to_colmap_model(
        colmap_model_dir,
        input_sparse_path,
        R=np.eye(3),
        t=np.zeros(3),
        scale=scale,
        output_sparse_path=output_sparse_path,
        qr_poses=qr_poses,
    )


def align_to_prior_map(colmap_model_dir, input_sparse_path, qr_poses, prior_poses, output_sparse_path=None):
    common_ids = set(prior_poses.keys()).intersection(set(qr_poses.keys()))
    if len(common_ids) < 2:
        print(f"!! Need at least 2 common QR IDs to compute sim3, got {len(common_ids)}")
        return

    prior_pts = []
    est_pts = []
    for qr_id in common_ids:
        prior_t = prior_poses[qr_id]["t"]
        est_t = np.array(qr_poses[qr_id]["t"], dtype=float)
        prior_pts.append(prior_t)
        est_pts.append(est_t)
        print(f"id: {qr_id}, prior t: {prior_t}, est t: {est_t}")
    prior_pts = np.array(prior_pts)
    est_pts = np.array(est_pts)

    def compute_sim3_transform(src_pts, dst_pts):
        assert len(src_pts) == len(dst_pts)
        src_pts = np.array(src_pts, dtype=np.float64)
        dst_pts = np.array(dst_pts, dtype=np.float64)

        src_centroid = np.mean(src_pts, axis=0)
        dst_centroid = np.mean(dst_pts, axis=0)

        src_centered = src_pts - src_centroid
        dst_centered = dst_pts - dst_centroid

        H = src_centered.T @ dst_centered / len(src_pts)

        # SVD
        U, S, Vt = np.linalg.svd(H)

        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.eye(3)
        D[2, 2] = d
        R = Vt.T @ D @ U.T

        var_src = np.sum(src_centered**2) / len(src_pts)
        scale = np.sum(S * np.diag(D)) / var_src

        t = dst_centroid - scale * R @ src_centroid

        return R, t, scale

    R, t, s = compute_sim3_transform(est_pts, prior_pts)
    print(f"Computed sim3 to prior: scale={s}, R:\n{R}, t:{t}")

    apply_sim3_to_colmap_model(
        colmap_model_dir,
        input_sparse_path,
        R=R,
        t=t,
        scale=s,
        output_sparse_path=output_sparse_path,
        qr_poses=qr_poses,
    )


def load_qr_json(qr_json_file):
    if qr_json_file and os.path.isfile(qr_json_file):
        with open(qr_json_file, "r", encoding="utf-8") as f:
            qr_json = json.load(f)
            qr_size_m = qr_json.get("qr_size_m", 0.174)
            qr_distances_raw = qr_json.get("qr_distances_m", None)
            qr_distances_m = {tuple(k.split(",")): float(v) for k, v in qr_distances_raw.items()}

            print(f"Loaded qr_size_m: {qr_size_m}, qr_distances_m: {qr_distances_m}")
            return qr_size_m, qr_distances_m

    print("No valid qr_json_file, use default qr_size_m=0.174m and no pairwise distances.")
    return 0.174, None


def load_qr_poses_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = {}
    for qr_id, pose in data.items():
        poses[qr_id] = {
            "R": np.array(pose["R"], dtype=float),
            "t": np.array(pose["t"], dtype=float),
        }

    return poses


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--colmap_model_dir", help="directory containing cameras.txt and images.txt")
    p.add_argument(
        "--image_dir",
        default="images",
        help="directory of input images (same names as in images.txt)",
    )
    p.add_argument("--qr_json_file", default="", help="input json file of qr distances and size")
    p.add_argument(
        "--prior_qr_poses_json_file",
        default="qr_poses_prior.json",
        help="input json file of prior qr poses (optional)",
    )
    p.add_argument("--output_dir", default="", help="output dir")
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

    qr_size_m, qr_distances_m = load_qr_json(args.qr_json_file)

    out = estimate_qrcode_poses_optimize(
        args.colmap_model_dir,
        args.input_sparse_path,
        args.image_dir,
        qr_size_m=qr_size_m,
        qr_distances_m=qr_distances_m,
        output_dir=args.output_dir if args.output_dir else None,
    )

    print("\n=== QR Code ESTIMATION ===")
    print(out["qr_poses"])
    print(out["edge_len_stats"])
    print(out["pairwise_dists_est"])

    print("\n=== SCALE ESTIMATION ===")
    print("candidates from size:", out["scale_info"]["scale_candidates"]["from_size"])
    print("candidates from pairs:", out["scale_info"]["scale_candidates"]["from_pairs"])
    print("final scale (median):", out["scale_info"]["scale_final"])
    print(
        "used size:",
        out["scale_info"]["used"]["size"],
        " used pairs:",
        out["scale_info"]["used"]["pairs"],
    )

    # Recover scale
    if args.input_sparse_path == args.output_sparse_path:
        print("input_sparse_path and output_sparse_path are the same, stop overwriting!")
    else:
        if args.output_sparse_path is not None:
            recover_scale(
                args.colmap_model_dir,
                args.input_sparse_path,
                out["qr_poses"]["anchors"],
                out["scale_info"]["scale_final"],
                args.output_sparse_path,
            )

    # if has prior qr poses, compute sim3 between prior and estimated, and align
    if args.prior_qr_poses_json_file:
        prior_file_path = os.path.join(args.colmap_model_dir, args.prior_qr_poses_json_file)
        if os.path.isfile(prior_file_path):
            print("\n=== SIM3 TO PRIOR QR POSES ===")
            prior_poses = load_qr_poses_from_json(prior_file_path)

            align_to_prior_map(
                args.colmap_model_dir,
                args.output_sparse_path if args.output_sparse_path else args.input_sparse_path,
                out["qr_poses"]["anchors"],
                prior_poses,
                output_sparse_path=args.output_sparse_path,
            )
