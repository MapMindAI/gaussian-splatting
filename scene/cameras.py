#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
from PIL import Image


class Camera(nn.Module):
    def read_image(self, image_path, resolution):
        image = Image.open(image_path)
        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        if resized_image_rgb.shape[0] == 4:
            alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else:
            alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
        
        if self.train_test_exp and self.is_test_view:
            if self.is_test_dataset:
                alpha_mask[..., :alpha_mask.shape[-1] // 2] = 0
            else:
                alpha_mask[..., alpha_mask.shape[-1] // 2:] = 0
        
        original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)

        return alpha_mask, original_image


    def load_image(self):
        if not self.dynamic_memory:
            return
        if self.original_image is None:
            self.alpha_mask, self.original_image = self.read_image(self.image_path, self.resolution)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]


    def unload_image(self):
        if self.dynamic_memory:
            self.alpha_mask = None
            self.original_image = None


    def get_invdepth(self):
        invdepthmap_full = cv2.resize(self.invdepthmap, self.resolution)
        invdepthmap_full = torch.from_numpy(invdepthmap_full[None]).to(self.data_device)
        return invdepthmap_full

    
    def get_depth_mask(self):
        depth_mask_full = cv2.resize(self.depth_mask, self.resolution)
        depth_mask_full = torch.from_numpy(depth_mask_full[None]).to(self.data_device)
        return depth_mask_full
    
            
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image_path, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 dynamic_memory = False):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.train_test_exp = train_test_exp
        self.is_test_dataset = is_test_dataset
        self.is_test_view = is_test_view
        self.dynamic_memory = dynamic_memory
        self.image_path = image_path
        self.resolution = resolution

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if dynamic_memory:
            self.alpha_mask = None
            self.original_image = None
        else:
            self.alpha_mask, self.original_image = self.read_image(self.image_path, self.resolution)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]      
        
        depth_buffer_resolution = 518  # process size for depth anything
        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            # self.depth_mask = torch.ones_like(self.alpha_mask)
            self.depth_mask = np.ones((resolution[1], resolution[0]))
            self.invdepthmap = cv2.resize(invdepthmap, self.resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True
            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            
            self.invdepthmap = cv2.resize(self.invdepthmap, (depth_buffer_resolution, depth_buffer_resolution))
            self.depth_mask = cv2.resize(self.depth_mask, (depth_buffer_resolution, depth_buffer_resolution))
            
            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask = None
                elif depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]
                    # mask for far depth
                    self.depth_mask[self.invdepthmap > 15.0] = 0
                    
            # self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
