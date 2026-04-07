import sqlite3
import argparse
import sys
import os
import numpy as np
import warnings
from collections import Counter


# 忽略所有警告
warnings.filterwarnings("ignore")

sys.path.append("scene")
sys.path.append("submodules/UTM")
from colmap_loader import *
from utm import LatLonToUTMXY


def gps_to_utm(lat, lon):
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    # from pyproj import Transformer, CRS
    # transformer = Transformer.from_crs("EPSG:4326", f"EPSG:326{utm_zone}" if hemisphere == 'north' else f"EPSG:327{utm_zone}", always_xy=True)
    # easting, northing = transformer.transform(lon, lat)
    easting, northing = LatLonToUTMXY(lat, lon, utm_zone)
    return easting, northing, utm_zone, hemisphere


def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


# https://github.com/colmap/colmap/blob/6556b4e28fba070e15894833b31f66de6cf4c6e1/scripts/python/database.py#L144
def read_images_gps_prior(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # tables = cursor.fetchall()
    # print("Tables contained in the database: ")
    # for table in tables:
    #     print("  - " + table[0])

    image_gpses = {}
    # take out GPS prior from database
    cursor.execute("SELECT * FROM pose_priors;")
    pose_priors = cursor.fetchall()
    for pose_prior in pose_priors:
        # img_id, pos, _, _ = pose_prior
        image_gpses[pose_prior[0]] = blob_to_array(pose_prior[1], np.float64, (3, 1))

    conn.close()
    return image_gpses


def trajectory_align_umeyama(
    trajectory_reference, trajectory
):
    """Align the two trajectory: trajectory_reference = s * R * trajectory + t
    Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    Args:
        trajectory_reference (array): 3xn array of the reference trajectory.
        trajectory (array): 3xn array of the trajectory to fit.
        scale_one (bool): Whether the default scale is one, or need estimation.
        yaw_only (bool): Whether the rotation is only yaw or not.

    Raises:
        AttributeError: trajectory_reference, trajectory is required.

    Return:
        tuple: estimation of the model: error = trajectory_reference - s * R * trajectory + t
            (
                scale,
                rotation,
                translation,
                error
            )
    """
    assert trajectory_reference.shape[0] == 3
    assert trajectory.shape[0] == 3
    assert trajectory_reference.shape[1] == trajectory.shape[1]

    # substract mean
    centroid_reference = trajectory_reference.mean(axis=1).reshape([3, 1])
    centroid = trajectory.mean(axis=1).reshape([3, 1])

    normalized_reference = trajectory_reference - centroid_reference
    normalized = trajectory - centroid
    num_pts = trajectory_reference.shape[1]

    # correlation
    tmp = np.dot(normalized_reference, normalized.transpose())
    correlation = 1.0 / num_pts * tmp
    sigma_sqr = 1.0 / num_pts * np.multiply(normalized, normalized).sum()
    u_svd, diag_d_svd, vt_svd = np.linalg.svd(correlation)
    d_svd = np.diag(diag_d_svd)
    v_svd = np.transpose(vt_svd)

    s_mat = np.eye(3)
    if np.linalg.det(u_svd) * np.linalg.det(v_svd) < 0:
        s_mat[2, 2] = -1
    rotation_estimate = np.dot(u_svd, np.dot(s_mat, np.transpose(v_svd)))
    scale = 1.0 / sigma_sqr * np.trace(np.dot(d_svd, s_mat))

    translation_estimate = centroid_reference - scale * np.dot(
        rotation_estimate, centroid
    )
    error = trajectory_reference - (
        scale * np.dot(rotation_estimate, trajectory) + translation_estimate
    )
    return scale, rotation_estimate, translation_estimate, np.linalg.norm(error, axis=0)


def robust_trajectory_alignment(trajectory_reference, trajectory, num_bin=10):
    scale, rotation, translation, errors = trajectory_align_umeyama(trajectory_reference, trajectory)
    hist, bin_edges = np.histogram(errors, bins=num_bin)  # 30 bins
    print("  - Histogram values:", hist)
    print("  - Bin edges:", bin_edges)
    for i in range(len(bin_edges)):
        bin_idx = i
        if bin_edges[i] < 5.0:
            continue
        break
    if bin_idx == 0:
        bin_idx = 1

    print(f"  - Bin edges: {bin_edges}")
    print(f"  - choose {bin_idx}th bin, with error {bin_edges[bin_idx]}")
    best_inliers = errors < bin_edges[1]
    # use all the inliers to recompute the result
    reference_subset = trajectory_reference[:, best_inliers]
    trajectory_subset = trajectory[:, best_inliers]

    if reference_subset.shape[1] < 3:
        print("Not enough GPS points")
        return 1.0, np.eye(3), np.zeros((3, 1)), []

    # Estimate the transformation using the subset
    scale, rotation, translation, error_part = trajectory_align_umeyama(reference_subset, trajectory_subset)
    return scale, rotation, translation, best_inliers


def filterPointsOnce(data, threshold):
    # filter points which are too far or isolated
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scores = np.abs((data - mean) / std)
    mask = np.all(z_scores < threshold, axis=1)
    return mask


def filterPointsIteration(data, iter=3, threshold=5):
    mask_final = np.ones(data.shape[0], dtype=bool)
    for _ in range(iter):
        mean = np.mean(data[mask_final, :], axis=0)
        std = np.std(data[mask_final, :], axis=0)

        z_scores = np.abs((data - mean) / std)
        mask = np.all(z_scores < threshold, axis=1)
        mask_final = mask_final & mask
    return mask_final


def filerUtmZone(utm_zones):
    # Count the occurrences of each number
    count = Counter(utm_zones)
    # Find the number with the highest count
    most_common_number = count.most_common(1)[0][0]
    return np.array(utm_zones) == most_common_number, most_common_number


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', help='input database path', type=str)
    parser.add_argument('--model_path', help='input model path', type=str)
    parser.add_argument('--output_model_path', help='output model path', type=str)
    args = parser.parse_args()
    return args

def log_print(f_log, *args, **kwargs):
    print(*args, **kwargs)          # print on screen
    print(*args, **kwargs, file=f_log)  # write to file


"""
python dm/colmap/transform_colmap_model.py \
--database_path /mnt/data/yeliu/gaussian_splatting/DJI_test/database.db \
--model_path /mnt/data/yeliu/gaussian_splatting/DJI_test/sparse_raw/0 \
--output_model_path /mnt/data/yeliu/gaussian_splatting/DJI_test/sparse/0

python dm/colmap/transform_colmap_model.py \
--database_path data/SYU/database.db \
--model_path data/SYU/sparse_raw/0 \
--output_model_path data/SYU/sparse_gps/0
"""
if __name__ == "__main__":
    args = parse_args();
    ""
    log_path = os.path.join(args.output_model_path, "transform_log.txt")
    os.makedirs(args.output_model_path, exist_ok=True)
    with open(log_path, "w") as f_log:

        image_gpses = read_images_gps_prior(args.database_path)

        # read the colmap model
        images, cameras, points3D = read_colmap_model(args.model_path)

        points_gps = []
        points_sfm = []
        utm_zones = []

        tmp = 0
        for image_id in images:
            if not image_id in image_gpses:
                continue
            # get the camera position
            image = images[image_id]
            camera_position = -np.dot(qvec2rotmat(image.qvec).transpose(), np.transpose(image.tvec))
            points_sfm.append(camera_position)

            gps = image_gpses[image_id]
            utm_easting, utm_northing, utm_zone, utm_letter = gps_to_utm(gps[0], gps[1])
            points_gps.append([utm_easting, utm_northing, gps[2]])
            utm_zones.append(utm_zone)

        # all the points should be in one zone (cross zone collection is rare)
        mask_utm, utm_zone = filerUtmZone(utm_zones)
        points_gps = np.array(points_gps).astype(float)[mask_utm, :]
        points_sfm = np.array(points_sfm).astype(float)[mask_utm, :]

        # filter GPS data
        mask = filterPointsIteration(points_gps, 1, 2)
        gps_mean = np.mean(points_gps[mask, :], axis=0)
        points_gps = np.transpose(points_gps[mask, :] - gps_mean)
        points_sfm = np.transpose(points_sfm[mask, :])
        radius = np.mean(np.linalg.norm(points_gps, axis=0))
        log_print(f_log, f"  - find {points_gps.shape[1]} / {len(utm_zones)} pairs, in UTM {utm_zone}, radius : {radius}")
        log_print(f_log, f"  - gps_mean: {gps_mean}")

        if points_gps.shape[1] < 5:
            log_print(f_log, "Not enough GPS points")
            # exit(1)
            rotation = np.eye(3)
            scale = 1.0
            translation = np.zeros((3, 1))
        else:
            # trajectory_reference = s * R * trajectory + t
            scale, rotation, translation, errors = robust_trajectory_alignment(points_gps, points_sfm)

            if scale > 1e4 or scale < 0.1:
                log_print(f_log, "[WARNING] STRANGE SCALE ERROR : ", scale)
                rotation = np.eye(3)
                scale = 1.0

            # read the colmap model
            log_print(f_log, f"  - estimated transformation:\n  scale = {scale}\n  rotation =\n{rotation}\n  translation = {translation.reshape(-1).tolist()}")
            translation = np.zeros((3, 1))
            log_print(f_log, f"But use (0, 0, 0) as translation currently!")

        # make the new model to output_model_path
        new_images = {}
        for image_id in images:
            image = images[image_id]
            camera_to_world_rot = qvec2rotmat(image.qvec).transpose()
            camera_to_world_trans = -np.dot(camera_to_world_rot, image.tvec.reshape((3, 1)))

            utm_to_camera_rot = np.transpose(np.dot(rotation, camera_to_world_rot))
            utm_to_camera_trans = -np.dot(utm_to_camera_rot, scale * np.dot(rotation, camera_to_world_trans) + translation)
            # print(scale * np.dot(rotation, camera_to_world_trans) + translation)
            new_images[image_id] = Image(
                id=image.id, qvec=rotmat2qvec(utm_to_camera_rot), tvec=utm_to_camera_trans.reshape(3),
                camera_id=image.camera_id, name=image.name,
                xys=image.xys, point3D_ids=image.point3D_ids)

        new_points3D = {}
        for point3D_id in points3D:
            point = points3D[point3D_id]
            new_xyz = scale * np.dot(rotation, point.xyz.reshape((3, 1))) + translation
            new_points3D[point3D_id] = Point3D(
                id=point.id,
                xyz=new_xyz.reshape(3),
                rgb=point.rgb,
                error=point.error,
                image_ids=point.image_ids,
                point2D_idxs=point.point2D_idxs,
            )

        # save the new model
        write_images_text(new_images, os.path.join(args.output_model_path, "images.txt"))
        write_cameras_text(cameras, os.path.join(args.output_model_path, "cameras.txt"))
        write_points3D_text(new_points3D, os.path.join(args.output_model_path, "points3D.txt"))
        write_images_binary(new_images, os.path.join(args.output_model_path, "images.bin"))
        write_cameras_binary(cameras, os.path.join(args.output_model_path, "cameras.bin"))
        write_points3D_binary(new_points3D, os.path.join(args.output_model_path, "points3D.bin"))

        log_print(f_log, "Done!")
