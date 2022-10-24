#!/usr/bin/env python3
#
# File   : utils.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
#
# Copyright 2022 Adobe. All rights reserved.
#
# This file is licensed to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR REPRESENTATIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import itertools
from typing import Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

from dycheck import geometry
from dycheck.utils import common, image


def rotate(
    img: np.ndarray,
    rotate_mode: Literal[
        "clockwise_0", "clockwise_90", "clockwise_180", "clockwise_270"
    ],
):
    if rotate_mode == "clockwise_0":
        return img
    return cv2.rotate(
        img,
        {
            "clockwise_90": cv2.ROTATE_90_CLOCKWISE,
            "clockwise_180": cv2.ROTATE_180,
            "clockwise_270": cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[rotate_mode],
    )


def rotate_intrin(
    K: np.ndarray,
    img_wh: Tuple[int, int],
    rotate_mode: Literal[
        "clockwise_0", "clockwise_90", "clockwise_180", "clockwise_270"
    ],
):
    W, H = img_wh
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, -1], K[1, -1]
    if rotate_mode == "clockwise_0":
        pass
    elif rotate_mode == "clockwise_90":
        K = np.array(
            [[fy, 0, H - 1 - cy], [0, fx, cx], [0, 0, 1]], dtype=K.dtype
        )
        img_wh = img_wh[::-1]
    elif rotate_mode == "clockwise_180":
        K = np.array(
            [[fx, 0, W - 1 - cx], [0, fy, H - 1 - cy], [0, 0, 1]],
            dtype=K.dtype,
        )
    elif rotate_mode == "clockwise_270":
        K = np.array(
            [[fy, 0, cy], [0, fx, W - 1 - cx], [0, 0, 1]], dtype=K.dtype
        )
        img_wh = img_wh[::-1]
    else:
        raise ValueError(rotate_mode)
    return K, img_wh


def rotate_transfm(
    rotate_mode: Literal[
        "clockwise_0", "clockwise_90", "clockwise_180", "clockwise_270"
    ]
):
    theta = float(rotate_mode.lstrip("clockwise_")) / 180 * np.pi
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        np.float32,
    )


def rotate_c2ws(
    c2ws: np.ndarray,
    rotate_mode: Literal[
        "clockwise_0", "clockwise_90", "clockwise_180", "clockwise_270"
    ],
):
    transfm = rotate_transfm(rotate_mode)
    btransm = np.broadcast_to(transfm, c2ws.shape[:-2] + (3, 3))
    c2ws = c2ws.copy()
    c2ws[..., :3, :3] = btransm @ c2ws[..., :3, :3] @ btransm.swapaxes(-2, -1)
    c2ws[..., :3, -1] = (btransm @ c2ws[..., :3, -1:])[..., 0]
    return c2ws


def sobel_by_quantile(img_points: np.ndarray, q: float):
    """Return a boundary mask where 255 indicates boundaries (where gradient is
    bigger than quantile).
    """
    dx0 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[1:-1, :-2], axis=-1
    )
    dx1 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[1:-1, 2:], axis=-1
    )
    dy0 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[:-2, 1:-1], axis=-1
    )
    dy1 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[2:, 1:-1], axis=-1
    )
    dx01 = (dx0 + dx1) / 2
    dy01 = (dy0 + dy1) / 2
    dxy01 = np.linalg.norm(np.stack([dx01, dy01], axis=-1), axis=-1)

    # (H, W, 1) uint8
    boundary_mask = (dxy01 > np.quantile(dxy01, q)).astype(np.float32)
    boundary_mask = (
        np.pad(boundary_mask, ((1, 1), (1, 1)), constant_values=False)[
            ..., None
        ].astype(np.uint8)
        * 255
    )
    return boundary_mask


def dilate(img: np.ndarray, kernel_size: Optional[int]):
    if kernel_size is None:
        return img
    is_float = np.issubdtype(img.dtype, np.floating)
    img = image.to_uint8(img)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    if is_float:
        dilated = image.to_float32(dilated)
    return dilated


def tsdf_fusion(
    imgs: np.ndarray,
    depths: np.ndarray,
    cameras: Sequence[geometry.Camera],
    *,
    voxel_length: float = 1,
    sdf_trunc: float = 0.01,
    depth_far: float = 1e5,
):
    import open3d as o3d

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for rgb, depth, camera in zip(
        common.tqdm(imgs, desc="* Fusing RGBDs"),
        depths,
        cameras,
    ):
        if (depth != 0).sum() == 0:
            continue
        # Make sure that the RGBD image is contiguous.
        rgb = o3d.geometry.Image(np.array(rgb))
        depth = o3d.geometry.Image(np.array(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1,
            depth_trunc=depth_far,
            convert_rgb_to_intensity=False,
        )
        w2c = camera.extrin
        W, H = camera.image_size
        fx = fy = camera.focal_length
        cx, cy = camera.principal_point
        volume.integrate(
            rgbd, o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy), w2c
        )

    pcd = volume.extract_point_cloud()
    return np.asarray(pcd.points), np.asarray(pcd.colors)


def get_bbox_segments(bbox: np.ndarray):
    points = x000, x001, x010, x011, x100, x101, x110, x111 = np.array(
        list(itertools.product(*bbox.T.tolist()))
    )
    end_points = [x001, x011, x000, x010, x101, x111, x100, x110]
    points = points.tolist()
    points += [x000, x001, x010, x011]
    end_points += [x100, x101, x110, x111]

    return np.array(points), np.array(end_points)
