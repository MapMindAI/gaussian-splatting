import os
import sys
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

sys.path.append("scene")
from colmap_loader import *


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def colmap_to_opengl(pose_qvec, pose_tvec):
    # COLMAP gives world-to-camera: [R | t]
    # Open3D expects camera-to-world
    R_wc = qvec2rotmat(pose_qvec)
    T = np.eye(4)
    T[:3, :3] = R_wc
    T[:3, 3] = pose_tvec
    return T


def load_colmap_model(colmap_dir):
    images, cameras, _ = read_colmap_model(colmap_dir)
    return images, cameras


def pose_process_mesh(mesh, min_cluster_triangles = 200):
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()

    # Remove very small triangle clusters
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()

    # Identify clusters to keep
    triangle_mask = [
        cluster_n_triangles[cluster_id] > min_cluster_triangles
        for cluster_id in triangle_clusters
    ]

    mesh.remove_triangles_by_mask(np.logical_not(triangle_mask))
    mesh.remove_unreferenced_vertices()

    return mesh


def integrate_tsdf(colmap_dir, rgb_dir, depth_dir, voxel_length=0.2, sdf_trunc=0.5):
    images, cameras = load_colmap_model(colmap_dir)

    # Initialize TSDF volume
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    progress_bar = tqdm(range(0, len(images)), desc="TSDF")
    for image_id, image in images.items():
        img_name = image.name
        rgb_path = os.path.join(rgb_dir, img_name)
        depth_path = os.path.join(depth_dir, img_name[:-4] + ".depth")

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(" ", rgb_path, "or", depth_path, " not found")
            continue

        progress_bar.set_postfix({"IMG": f"{img_name}"})
        progress_bar.update(1)

        rgb_o3d = o3d.io.read_image(rgb_path)
        inv_depth = read_array(depth_path)
        inv_depth[inv_depth == 0] = 0.0001
        depth_np = (1000.0 / inv_depth).astype(np.uint16)
        depth_np = np.ascontiguousarray(depth_np)  # Ensure C-contiguous
        depth_o3d = o3d.geometry.Image(depth_np)

        cam = cameras[image.camera_id]
        assert cam.model is "PINHOLE"

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=cam.width,
            height=cam.height,
            fx=cam.params[0],
            fy=cam.params[1],
            cx=cam.params[2],
            cy=cam.params[3]
        )

        pose = colmap_to_opengl(image.qvec, image.tvec)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1000.0,
            depth_trunc=20.0,
            convert_rgb_to_intensity=False
        )

        tsdf_volume.integrate(rgbd, intrinsic, pose)

    progress_bar.close()
    print("Extracting mesh...")
    mesh = tsdf_volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print("Post process mesh...")
    mesh = pose_process_mesh(mesh)
    return mesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='input model path', type=str)
    parser.add_argument('--images', help='images path', default="images", type=str)
    parser.add_argument('--sparse', help='sparse path', default="sparse/0", type=str)
    parser.add_argument('--depths', help='depths path', default="output/depth", type=str)
    parser.add_argument('--viewer', help='run viewer', default=0, type=int)
    args = parser.parse_args()
    return args


# Example usage: python dm/tsdf/tsdf_modeling.py --model_path /mnt/ml-experiment-data/yeliu/gaussian_splatting/GoPro/NanshaOffice
if __name__ == '__main__':
    args = parse_args()

    colmap_dir = os.path.join(args.model_path, args.sparse)
    rgb_dir = os.path.join(args.model_path, args.images)
    depth_dir = os.path.join(args.model_path, args.depths)
    mesh_path = os.path.join(args.model_path, "output/tsdf_mesh")
    mesh = integrate_tsdf(colmap_dir, rgb_dir, depth_dir)

    print("Save mesh...")
    o3d.io.write_triangle_mesh(mesh_path + ".ply", mesh)
    o3d.io.write_triangle_mesh(mesh_path + ".obj", mesh)
    if args.viewer == 1:
        o3d.visualization.draw_geometries([mesh])
