import sqlite3
import numpy as np
import struct
import shutil
import argparse
import os
import cv2
import open3d
import torch
from tqdm import tqdm

import sys

sys.path.append("scene")
sys.path.append("submodules")
from colmap_loader import *

from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional
from LightGlue.lightglue import SuperPoint

FEATURE_TYPE=4

class COLMAPDatabase:
    def __init__(self, original_db_path: str, new_db_path: str = None):
        """
        Initialize COLMAP database connection

        Args:
          original_db_path: Original database file path
          new_db_path: New database file path (if None, will create based on original name)
        """
        self.original_db_path = original_db_path
        self.new_db_path = new_db_path
        self.reconstruction_path = original_db_path.replace("database.db", "sparse/0/")
        self.max_image_width = 1080

        # Copy original database to new path
        shutil.copy2(original_db_path, self.new_db_path)
        print(f"Copied database from {original_db_path} to {self.new_db_path}")

        # Connect to the new database
        self.conn = sqlite3.connect(self.new_db_path)
        self.cursor = self.conn.cursor()

        # Clear existing matches
        self.cursor.execute("DROP TABLE IF EXISTS matches")
        self.cursor.execute("DROP TABLE IF EXISTS pose_priors")
        self.cursor.execute("DROP TABLE IF EXISTS two_view_geometries")
        self.conn.commit()

        # Write image pose to the database
        self.update_image_table_with_poses()

        # add 3d_coords and colors column
        self.add_3d_coords_column()

        # If use_superpoint, replace original features
        self.replace_with_superpoint()

        self.print_database_info()

    def replace_with_superpoint(self):
        """
        Replace original features (keypoints and descriptors) with superpoint features
        """

        print(f"-----------Replacing original features with superpoint features-----------")
        # 1. Get superpoint and load all image_id
        superpoint = SuperPoint(max_num_keypoints=4096).eval().to("cuda")
        image_ids = self.get_all_image_ids()
        success_count = 0
        print(f"Get superpoint and load all image_id!")

        # 2. Clear existing keypoints and descriptors
        self.cursor.execute("DELETE FROM keypoints")
        self.cursor.execute("DELETE FROM descriptors")
        self.conn.commit()
        print(f"Clear existing keypoints and descriptors!")

        # 3. Extract superpoint features in all images and update keypoints and descriptors table
        progress_bar = tqdm(range(0, len(image_ids)), desc="Extract Features")
        for image_id in image_ids:
            progress_bar.update(1)
            # print(f'Extracting features in image {image_id}...')
            # 3.1 Load image
            self.cursor.execute(
                """
        SELECT name FROM images WHERE image_id = ?
      """,
                (image_id,),
            )
            result = self.cursor.fetchone()
            if result is None:
                print(f"[Warning] Image ID {image_id} not found, skipping...\n")
                continue
            image_name = result[0]
            image_path = os.path.join(os.path.dirname(self.original_db_path), "images/" + image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[Warning] Failed to load image {image_path}, skipping...\n")
                continue

            # 3.2 Run superpoint
            try:
                # ktps, descps = superpoint.run(image)
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
                if image_gray.shape[1] > self.max_image_width:
                    # resize the image, if image size too large
                    new_height = int(self.max_image_width * image_gray.shape[0] / image_gray.shape[1])
                    image_size = (self.max_image_width, new_height)
                    image_gray = cv2.resize(image_gray, image_size).astype(np.uint8)
                image_tensor = torch.from_numpy(image_gray / 255.0).float()[None, None].to("cuda")
                image_size = (image_gray.shape[1], image_gray.shape[0])

                pred0 = superpoint({"image": image_tensor})
                old_ktps = pred0["keypoints"][0].detach().cpu().numpy()
                descps = pred0["descriptors"][0].detach().cpu().numpy()

                ktps = []
                factor_x = image.shape[1] / image_size[0]
                factor_y = image.shape[0] / image_size[1]
                for i in range(old_ktps.shape[0]):
                    ktps.append([factor_x * old_ktps[i][0], factor_y * old_ktps[i][1]])
                ktps = np.array([ktps], dtype=np.float32)

                # normalize descriptions
                descps = 255.0 * (descps + 0.5)
                descps = np.array([descps], dtype=np.uint8)

                # ktps, descps = ktps[0], descps[0]
            except Exception as e:
                print(f"[Error] Superpoint failed on {image_name}: {e}, skipping...\n")
                continue

            if ktps is None or descps is None or ktps.shape[0] == 0:
                print(f"[Info] No keypoints detected for {image_name}, skipping...\n")
                continue

            # 3.3 Extract color information at keypoint locations
            colors = []
            for i in range(ktps.shape[1]):
                x, y = int(ktps[0, i, 0]), int(ktps[0, i, 1])  # keypoint coordinates
                # Ensure coordinates are within image bounds
                y = max(0, min(y, image.shape[0] - 1))
                x = max(0, min(x, image.shape[1] - 1))
                # Get BGR color and convert to RGB
                bgr_color = image[y, x]
                rgb_color = [bgr_color[2], bgr_color[1], bgr_color[0]]  # BGR to RGB
                colors.append(rgb_color)
            colors = np.array(colors, dtype=np.uint8)

            # 3.4 Update keypoints and descriptors table
            num_kpts = ktps.shape[1]
            desc_dim = descps.shape[2]

            keypoints_blob = ktps.astype(np.float32).tobytes()
            colors_blob = colors.astype(np.uint8).tobytes()
            self.cursor.execute(
                "INSERT INTO keypoints (image_id, rows, cols, data, colors) VALUES (?, ?, ?, ?, ?)",
                (image_id, num_kpts, 2, keypoints_blob, colors_blob),
            )

            desc_blob = descps.astype(np.uint8).tobytes()
            self.cursor.execute(
                "INSERT INTO descriptors (image_id, rows, cols, data, type) VALUES (?, ?, ?, ?, ?)",
                (image_id, num_kpts, desc_dim, desc_blob, FEATURE_TYPE),
            )

            success_count += 1
            # print(f"Successfully updated image {image_id} with {num_kpts} superpoint keypoints\n")
        progress_bar.close()
        self.conn.commit()

        print(f"Total images successfully updated: {success_count} / {len(image_ids)}")
        print(f"------------------------------Replacing done!-----------------------------")

    def update_image_table_with_poses(self):
        """
        Fulfill poses to images table in database
        """

        # Get valid image names from depth files
        depth_root = self.original_db_path.replace("database.db", "output/depth/")
        valid_image_names = set()
        for root, _, files in os.walk(depth_root):
            for f in files:
                if f.endswith(".depth"):
                    # 获取相对路径（相对于 depth 根目录）
                    rel_path = os.path.relpath(os.path.join(root, f), depth_root)
                    # 替换扩展名
                    img_name = os.path.splitext(rel_path)[0] + ".jpg"
                    valid_image_names.add(img_name)

        print(f"Found {len(valid_image_names)} valid images from depth folder")

        # Delete images not in images.txt
        db_images = self.cursor.execute("SELECT image_id, name FROM images").fetchall()
        to_delete = [row[0] for row in db_images if row[1] not in valid_image_names]

        if to_delete:
            q_marks = ",".join(["?"] * len(to_delete))
            self.cursor.execute(f"DELETE FROM images WHERE image_id IN ({q_marks})", to_delete)
            self.cursor.execute(f"DELETE FROM keypoints WHERE image_id IN ({q_marks})", to_delete)
            self.cursor.execute(f"DELETE FROM descriptors WHERE image_id IN ({q_marks})", to_delete)
            print(f"Deleted {len(to_delete)} images not in images.txt")
        else:
            print("No extra images to delete")

        # Get poses of images from COLMAP reconstruction
        images, cameras, _ = read_colmap_model(self.reconstruction_path)
        print(f" load sparse model : #images={len(images)}, #cameras={len(cameras)}")

        poses = {}
        for image_id, image in images.items():
            # [qx, qy, qz, qw] in scipy as_quat()
            qvec = image.qvec
            tvec = image.tvec
            poses[image_id] = {
                "qw": qvec[0],
                "qx": qvec[1],
                "qy": qvec[2],
                "qz": qvec[3],
                "tx": tvec[0],
                "ty": tvec[1],
                "tz": tvec[2],
            }

        # TODO(yeliu): fix for all the cameras (there might be multiple cameras)
        # Get camera intrinsic from COLMAP reconstruction,
        # and update the cameras table with intrinsic
        camera = next(iter(cameras.values()))
        camera_id = camera.id
        params = camera.params
        self.cursor.execute(
            """
        UPDATE cameras
        SET params = ?
        WHERE camera_id = ?
    """,
            (params.tobytes(), camera_id),
        )

        # Update the images table with poses
        new_columns = ["qw", "qx", "qy", "qz", "tx", "ty", "tz"]
        for col in new_columns:
            try:
                self.cursor.execute(f"ALTER TABLE images ADD COLUMN {col} REAL")
            except sqlite3.OperationalError:
                # column exists, ignore
                pass

        for image_id, pose in poses.items():
            self.cursor.execute(
                """
        UPDATE images
        SET qw = ?, qx = ?, qy = ?, qz = ?, tx = ?, ty = ?, tz = ?
        WHERE image_id = ?
        """,
                (
                    pose["qw"],
                    pose["qx"],
                    pose["qy"],
                    pose["qz"],
                    pose["tx"],
                    pose["ty"],
                    pose["tz"],
                    image_id,
                ),
            )

        self.conn.commit()
        print("Image poses updated successfully!!!")

    def print_database_info(self):
        """
        Print basic information about the database, including table names and their structures.
        """
        # Print all tables in the database
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        print("\nTables in the database:")
        for table in tables:
            table_name = table[0]
            print(f"  Table: {table_name}")

            # Print table structure
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns = self.cursor.fetchall()
            print("  Columns:")
            for column in columns:
                print(f"    {column[1]}: {column[2]}")
            print(40 * "-")

    def get_database_path(self) -> str:
        """Get the path of the working database"""
        return self.new_db_path

    def add_3d_coords_column(self):
        """
        Add 3D coordinates column to keypoints table
        """
        try:
            # Check if coords_3d column already exists
            self.cursor.execute("PRAGMA table_info(keypoints)")
            columns = [column[1] for column in self.cursor.fetchall()]

            # TODO(wenhao): Let the column name become a variable parameter
            if "coords_3d" not in columns:
                # Add new column to store 3D coordinates (using BLOB to store numpy arrays)
                self.cursor.execute(
                    """
          ALTER TABLE keypoints
          ADD COLUMN coords_3d BLOB
        """
                )
                print("Successfully added coords_3d column to keypoints table")
            else:
                print("coords_3d column already exists")

            # Add colors column if it doesn't exist
            if "colors" not in columns:
                self.cursor.execute(
                    """
          ALTER TABLE keypoints
          ADD COLUMN colors BLOB
        """
                )
                print("Successfully added colors column to keypoints table")
            else:
                print("colors column already exists")

        except sqlite3.Error as e:
            print(f"Error adding column: {e}")

    def print_table_info(self, table_name: str):
        """
        Print information about the specified table

        Args:
          table_name: Name of the table to inspect
        """
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.cursor.fetchall()
        print(f"Columns in table '{table_name}':")
        for column in columns:
            print(f"  {column[1]}: {column[2]}")

    def get_camera_params(self, camera_id: int) -> Tuple[int, int, int, np.ndarray]:
        """
        Get camera parameters

        Args:
          camera_id: Camera ID

        Returns:
          (model, width, height, params) tuple
        """
        self.cursor.execute(
            """
      SELECT model, width, height, params FROM cameras WHERE camera_id = ?
    """,
            (camera_id,),
        )

        result = self.cursor.fetchone()
        if result is None:
            return None, None, None, None

        model, width, height, params_blob = result
        params = np.frombuffer(params_blob, dtype=np.float64)

        return model, width, height, params

    def get_image_info(self, image_id: int) -> Tuple[str, int, np.ndarray, np.ndarray]:
        """
        Get image information including pose

        Args:
          image_id: Image ID

        Returns:
          (name, camera_id, quat, trans) tuple
        """

        self.cursor.execute(
            """
      SELECT name, camera_id FROM images WHERE image_id = ?
    """,
            (image_id,),
        )
        result_image = self.cursor.fetchone()
        if result_image is None:
            return None, None, None, None

        name, camera_id = result_image

        self.cursor.execute(
            """
      SELECT qw, qx, qy, qz, tx, ty, tz FROM images WHERE image_id = ?
    """,
            (image_id,),
        )
        result_pose = self.cursor.fetchone()
        if result_pose is None:
            return None, None, None, None
        qw, qx, qy, qz, tx, ty, tz = result_pose
        quat = [qw, qx, qy, qz]
        trans = [tx, ty, tz]

        return name, camera_id, quat, trans

    def get_keypoints_data(self, image_id: int) -> np.ndarray:
        """
        Get keypoints data for specified image

        Args:
          image_id: Image ID

        Returns:
          Keypoints array with shape: (N, 6) [x, y, scale, orientation, octave, response]
        """
        self.cursor.execute(
            """
      SELECT rows, cols, data FROM keypoints WHERE image_id = ?
    """,
            (image_id,),
        )

        result = self.cursor.fetchone()
        if result is None:
            return np.array([])

        # COLMAP stores keypoints as float32 array
        rows, cols, keypoints_blob = result
        keypoints = np.frombuffer(keypoints_blob, dtype=np.float32)

        keypoints = keypoints.reshape(-1, cols)

        return keypoints

    def encode_3d_coords(self, coords_3d: np.ndarray) -> bytes:
        """
        Encode 3D coordinates array to binary format

        Args:
          coords_3d: 3D coordinates array with shape: (N, 3)

        Returns:
          Encoded binary data
        """
        if coords_3d.size == 0:
            return b""

        # Ensure float64 format and convert to bytes
        coords_3d = coords_3d.astype(np.float64)
        return coords_3d.tobytes()

    def decode_3d_coords(self, blob_data: bytes) -> np.ndarray:
        """
        Decode 3D coordinates binary data

        Args:
          blob_data: Binary data

        Returns:
          3D coordinates array with shape: (N, 3)
        """
        if not blob_data:
            return np.array([])

        coords = np.frombuffer(blob_data, dtype=np.float64)
        return coords.reshape(-1, 3)

    def decode_colors(self, blob_data: bytes) -> np.ndarray:
        """
        Decode colors binary data

        Args:
          blob_data: Binary data

        Returns:
          Colors array with shape: (N, 3) - RGB values
        """
        if not blob_data:
            return np.array([])

        colors = np.frombuffer(blob_data, dtype=np.uint8)
        return colors.reshape(-1, 3)

    def update_keypoints_3d_coords(self, image_id: int, coords_3d: np.ndarray):
        """
        Update 3D coordinates for keypoints of specified image

        Args:
          image_id: Image ID
          coords_3d: 3D coordinates array with shape: (N, 3)
        """
        encoded_coords = self.encode_3d_coords(coords_3d)

        self.cursor.execute(
            """
      UPDATE keypoints
      SET coords_3d = ?
      WHERE image_id = ?
    """,
            (encoded_coords, image_id),
        )

        if self.cursor.rowcount == 0:
            print(f"No keypoints data found for image {image_id}")

    def get_all_image_ids(self) -> List[int]:
        """Get all image IDs"""
        self.cursor.execute("SELECT DISTINCT image_id FROM keypoints")
        return [row[0] for row in self.cursor.fetchall()]

    def get_keypoints_with_3d_coords(self, image_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get keypoints and their 3D coordinates, and colors

        Args:
          image_id: Image ID

        Returns:
          (keypoints_2d, coords_3d, colors) tuple
        """
        self.cursor.execute(
            """
      SELECT rows, cols, data, coords_3d, colors FROM keypoints WHERE image_id = ?
    """,
            (image_id,),
        )

        result = self.cursor.fetchone()
        if result is None:
            return np.array([]), np.array([])

        # Decode 2D keypoints
        rows, cols, keypoints_blob, coords_3d_blob, colors_blob = result
        keypoints_2d = np.frombuffer(keypoints_blob, dtype=np.float32)

        keypoints_2d = keypoints_2d.reshape(-1, cols)

        # Decode 3D coordinates
        coords_3d = self.decode_3d_coords(coords_3d_blob) if coords_3d_blob else np.array([])

        # Decode colors
        colors = self.decode_colors(colors_blob) if colors_blob else np.array([])

        return keypoints_2d, coords_3d, colors

    def commit(self):
        """Commit changes"""
        self.conn.commit()

    def close(self):
        """Close database connection"""
        self.conn.close()


class DepthTo3DConverter:
    def __init__(self, depth_path: str, min_depth_percentile: float, max_depth_percentile: float):
        """
        Initialize depth to 3D converter

        Args:
          depth_path: Directory containing depth maps
        """
        self.depth_path = depth_path
        self.min_depth_percentile = min_depth_percentile
        self.max_depth_percentile = max_depth_percentile
        self.max_depth = 100.0

    def read_array(self, path):
        with open(path, "rb") as fid:
            width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
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

    def load_depth_map(self, depth_filename: str) -> Optional[np.ndarray]:
        """
        Load depth map from file

        Args:
          depth_filename: Depth map filename

        Returns:
          Depth map array or None if not found
        """
        depth_file_path = os.path.join(self.depth_path, depth_filename)

        if os.path.exists(depth_file_path):
            try:
                # TODO(wenhao): More suitable methods to remove outliers
                depth_map = self.read_array(depth_file_path)
                min_depth, max_depth = np.percentile(depth_map, [self.min_depth_percentile, self.max_depth_percentile])
                depth_map[depth_map < min_depth] = min_depth
                depth_map[depth_map > max_depth] = max_depth

                if depth_map is not None:
                    # print(f"Loaded depth map: {depth_file_path}")
                    return depth_map.astype(np.float32)
            except Exception as e:
                print(f"Error loading depth map {depth_file_path}: {e}")

        print(f"Could not find depth map for: {depth_file_path}")
        return None

    def pixel_to_3d(
        self,
        keypoints_2d: np.ndarray,
        depth_map: np.ndarray,
        camera_params: np.ndarray,
        camera_model: int,
        rotation_matrix: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        """
        Convert 2D keypoints to 3D coordinates using depth map

        Args:
          keypoints_2d: 2D keypoints array (N, 2) - [x, y] coordinates
          depth_map: Depth map array
          camera_params: Camera intrinsic parameters
          camera_model: Camera model type
          rotation_matrix: Camera rotation matrix (3x3), cam_to_world
          translation: Camera translation vector (3,), cam_to_world

        Returns:
          3D coordinates array (N, 3)
        """
        coords_3d = []

        # Extract intrinsic parameters (assuming PINHOLE model)
        assert camera_model is 1  # PINHOLE
        fx, fy, cx, cy = camera_params[:4]

        max_inv_depth = 1.0 / self.max_depth
        for i, kp in enumerate(keypoints_2d):
            x, y = kp[:2]  # Take first two coordinates

            # Convert to integer pixel coordinates
            px, py = int(round(x)), int(round(y))

            # Check if pixel is within depth map bounds
            if 0 <= px < depth_map.shape[1] and 0 <= py < depth_map.shape[0]:
                # Get depth value, inverse depth is stored inside the depth map
                depth_inv = depth_map[py, px]

                # Check if depth is valid (not zero or negative)
                if depth_inv > max_inv_depth and depth_inv < 1e6:
                    depth = 1.0 / depth_inv
                    # Convert pixel to normalized camera coordinates
                    x_norm = (x - cx) / fx
                    y_norm = (y - cy) / fy

                    # Convert to 3D camera coordinates
                    X_cam = x_norm * depth
                    Y_cam = y_norm * depth
                    Z_cam = depth

                    # Transform to world coordinates
                    point_cam = np.array([X_cam, Y_cam, Z_cam])
                    point_world = rotation_matrix @ point_cam + translation

                    coords_3d.append(point_world)
                else:
                    # Invalid depth, use NaN
                    coords_3d.append(np.array([np.nan, np.nan, np.nan]))
            else:
                # Out of bounds, use NaN
                coords_3d.append(np.array([np.nan, np.nan, np.nan]))

        return np.array(coords_3d)

    def process_image(self, db_tool: COLMAPDatabase, image_id: int) -> bool:
        """
        Process single image to compute 3D coordinates from depth

        Args:
          db_tool: Database tool instance
          image_id: Image ID to process

        Returns:
          True if successful, False otherwise
        """
        # Get image information, quat = [w, x, y, z]
        image_name, camera_id, quat, trans = db_tool.get_image_info(image_id)
        # print(f'name: {image_name}, id: {image_id}, quat: {quat}, trans: {trans}')

        if image_name is None:
            print(f"Could not find image info for ID: {image_id}")
            return False

        # Get camera parameters
        camera_model, width, height, camera_params = db_tool.get_camera_params(camera_id)
        if camera_model is None:
            print(f"Could not find camera info for ID: {camera_id}")
            return False

        # Get keypoints
        keypoints = db_tool.get_keypoints_data(image_id)
        if len(keypoints) == 0:
            print(f"No keypoints found for image ID: {image_id}")
            return False

        # Load depth map
        depth_map_name = image_name.replace(".jpg", ".depth")
        depth_map = self.load_depth_map(depth_map_name)

        if depth_map is None:
            print(f"Could not load depth map for image: {image_name}")
            return False
        # print(f"depth_map shape: {depth_map.shape}")

        # Convert quaternion to rotation matrix
        R_cw = qvec2rotmat(quat)
        t_cw = np.array(trans).reshape(3, 1)
        R_wc = R_cw.T
        C_w = -R_cw.T @ t_cw

        rotation_matrix = R_wc
        trans = C_w.flatten()

        # Convert 2D keypoints to 3D coordinates
        keypoints_2d = keypoints[:, :2]  # Extract x, y coordinates
        coords_3d = self.pixel_to_3d(keypoints_2d, depth_map, camera_params, camera_model, rotation_matrix, trans)

        # Update database
        db_tool.update_keypoints_3d_coords(image_id, coords_3d)

        # Count valid 3D points
        valid_points = np.sum(~np.isnan(coords_3d[:, 0]))
        # print(f"Image {image_name}: {valid_points}/{len(coords_3d)} keypoints have valid 3D coordinates")

        return True

    def create_ply_file(self, db_tool: COLMAPDatabase, file_name: str):
        """Add all 3d points from images together and create a .ply file"""
        point_cloud = open3d.geometry.PointCloud()
        all_points = []
        all_colors = []
        has_colors = False

        image_ids = db_tool.get_all_image_ids()
        progress_bar = tqdm(range(0, len(image_ids)), desc="Merge Pcl")

        for image_id in image_ids:
            _, point3D, colors = db_tool.get_keypoints_with_3d_coords(image_id)

            # Filter out invalid points (NaN values)
            valid_mask = ~np.isnan(point3D).any(axis=1)
            valid_points = point3D[valid_mask]

            if len(valid_points) > 0:
                all_points.extend(valid_points)

                # Check if colors are available
                if colors.size > 0:
                    valid_colors = colors[valid_mask]
                    all_colors.extend(valid_colors)
                    has_colors = True
            progress_bar.set_postfix({"#Pts": f"{len(valid_points)}", "Id": f"{image_id}"})
            progress_bar.update(1)
        progress_bar.close()

        if len(all_points) > 0:
            # Convert to numpy arrays
            all_points = np.array(all_points)
            point_cloud.points = open3d.utility.Vector3dVector(all_points)

            # Add colors if available
            if has_colors and len(all_colors) > 0:
                all_colors = np.array(all_colors)
                # Normalize colors to [0, 1] range for Open3D
                all_colors = all_colors.astype(np.float64) / 255.0
                point_cloud.colors = open3d.utility.Vector3dVector(all_colors)
                print(f"Point cloud created with {len(all_points)} points and colors")
            else:
                print(f"Point cloud created with {len(all_points)} points (no colors)")

            open3d.io.write_point_cloud(file_name, point_cloud)
            print(f"Point cloud saved as {file_name}")
        else:
            print("No valid points found, cannot create point cloud")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Add 3D coordinates to COLMAP keypoints using depth maps")
    parser.add_argument(
        "--map_dir",
        type=str,
        default="/mnt/ml-experiment-data/yeliu/gaussian_splatting/GoPro/NanshaOffice2",
        help="Directory containing COLMAP reconstruction (with database.db)",
    )
    parser.add_argument(
        "--new_db_name",
        type=str,
        default="database_3d.db",
        help="Name for the new database file",
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        default="output/depth",
        help="Directory containing depth maps",
    )
    parser.add_argument(
        "--min_depth_percentile",
        type=float,
        default=5,
        help="minimum visualization depth percentile",
    )
    parser.add_argument(
        "--max_depth_percentile",
        type=float,
        default=95,
        help="maximum visualization depth percentile",
    )
    parser.add_argument(
        "--whether_use_superpoint",
        type=int,
        default=1,
        help="whether use sp to replace keypoints table",
    )

    args = parser.parse_args()

    # Construct paths
    original_db_path = os.path.join(args.map_dir, "database.db")
    new_db_path = os.path.join(args.map_dir, args.new_db_name)
    depth_path = os.path.join(args.map_dir, args.depth_path)
    min_depth_percentile = args.min_depth_percentile
    max_depth_percentile = args.max_depth_percentile
    use_superpoint = args.whether_use_superpoint

    # Check if original database exists
    if not os.path.exists(original_db_path):
        print(f"Error: Original database not found at {original_db_path}")
        return

    # Check if depth directory exists
    if not os.path.exists(depth_path):
        print(f"Error: Depth directory not found at {depth_path}")
        return

    print(f"Original database: {original_db_path}")
    print(f"New database: {new_db_path}")
    print(f"Depth maps directory: {depth_path}")

    # Initialize tools
    if use_superpoint:
        db_tool = COLMAPDatabase(original_db_path, new_db_path)
    else:
        db_tool = COLMAPDatabase(original_db_path, new_db_path)

    depth_converter = DepthTo3DConverter(depth_path, min_depth_percentile, max_depth_percentile)

    try:
        # Process all images
        image_ids = db_tool.get_all_image_ids()
        print(f"Found {len(image_ids)} images to process")
        progress_bar = tqdm(range(0, len(image_ids)), desc="Load Point3d")

        successful_count = 0
        for image_id in image_ids:
            # print(f"\nProcessing image ID: {image_id}")
            if depth_converter.process_image(db_tool, image_id):
                successful_count += 1
            progress_bar.update(1)
        progress_bar.close()

        # Commit changes
        db_tool.commit()
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful_count}/{len(image_ids)} images")
        print(f"New database saved as: {new_db_path}")

        # Create point cloud
        ply_file_name = os.path.join(args.map_dir, "output/point_cloud_from_depth.ply")
        depth_converter.create_ply_file(db_tool, ply_file_name)

    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        db_tool.close()


if __name__ == "__main__":
    main()
