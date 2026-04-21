import numpy as np

# https://github.com/adynathos/panorama_to_pinhole/blob/master/panorama_to_pinhole.py
# Panorama sampling
from scipy.interpolate import RegularGridInterpolator


def panorama_to_sampler(panorama_img):
    h, w, _ = panorama_img.shape

    sampler = RegularGridInterpolator(
        [np.linspace(-0.5 * np.pi, 0.5 * np.pi, h), np.linspace(-np.pi, np.pi, w)],
        panorama_img,
        "linear",
    )

    return sampler


# cache because these grids only depend on camera intrinsics, so should rarely change
from functools import lru_cache


@lru_cache(maxsize=16)
def make_grid(w_h):
    w, h = w_h
    return np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=2)


@lru_cache(maxsize=16)
def make_img_pts(w_h, focal):
    img_coord_pix_g = make_grid(w_h)
    img_coord_pix = img_coord_pix_g.reshape(-1, 2)

    img_coord = (img_coord_pix - np.array(w_h) // 2) * (1.0 / focal)

    return img_coord


@lru_cache(maxsize=16)
def make_img_unit_vectors(w_h, focal):
    img_pts = make_img_pts(w_h, focal)

    # add focal as 3rd dimension, units such that f=1
    img_pts = np.concatenate([img_pts, np.ones((img_pts.shape[0], 1), dtype=img_pts.dtype)], axis=1)

    # normalize to put it on r=1 sphere
    img_pts /= np.linalg.norm(img_pts, axis=1)[:, None]

    return img_pts


# Rotation


def rot_around_x(angle):
    """rotation matrix around x axis"""
    s = np.sin(angle)
    c = np.cos(angle)

    return np.array(
        [
            [1, 0, 0],
            [0, c, s],
            [0, -s, c],
        ],
        dtype=np.float64,
    )


def rot_around_y(angle):
    """rotation matrix around y axis"""
    s = np.sin(angle)
    c = np.cos(angle)

    return np.array(
        [
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c],
        ],
        dtype=np.float64,
    )


def rot_around_z(angle):
    """rotation matrix around z axis"""
    s = np.sin(angle)
    c = np.cos(angle)

    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def R_cam_to_world(pitch, yaw, roll):
    R = np.identity(3)

    if yaw != 0:
        R = R @ rot_around_y(-yaw)

    if pitch != 0:
        R = R @ rot_around_x(-pitch)

    if roll != 0:
        R = R @ rot_around_z(-roll)

    return R


# Spherical coords


def unit_vectors_to_spherical(unit_vectors):
    # X = left right
    # Y = up down
    # Z = forward

    # pitch = asin(Y)
    sph_pitch = np.arcsin(unit_vectors[:, 1])

    # yaw = atan(y=x, x=z)
    sph_yaw = np.arctan2(unit_vectors[:, 0], unit_vectors[:, 2])  #

    return np.stack([sph_yaw, sph_pitch], axis=1)


def sample_spherical(sampler, sph_pts, img_w_h):
    colors = sampler(sph_pts[:, ::-1])
    colors = np.rint(colors).astype(np.uint8)
    img = colors.reshape(img_w_h[::-1] + (3,))
    return img


def panorama_to_pinhole(panorama_sampler, img_w_h, focal, pitch_yaw_roll_deg):
    unit_vs = make_img_unit_vectors(img_w_h, focal)

    # rotate
    rot_angles_rad = np.deg2rad(np.array(pitch_yaw_roll_deg))
    cam_R = R_cam_to_world(*rot_angles_rad)
    unit_vs_rotated = unit_vs @ cam_R.T

    # to spherical
    spherical = unit_vectors_to_spherical(unit_vs_rotated)

    # sample
    return sample_spherical(panorama_sampler, spherical, img_w_h)
