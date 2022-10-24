#!/usr/bin/env python3
#
# File   : __init__.py
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

from copy import deepcopy
from typing import Callable, Optional, Union

import cv2
import numpy as np

from dycheck.utils import image

from .skeleton import SKELETON_MAP, Skeleton


def visualize_kps(
    kps: np.ndarray,
    img: np.ndarray,
    *,
    skeleton: Union[str, Skeleton, Callable[..., Skeleton]] = "unconnected",
    rgbs: Optional[np.ndarray] = None,
    kp_radius: int = 4,
    bone_thickness: int = 3,
    **kwargs,
) -> np.ndarray:
    """Visualize 2D keypoints with their skeleton.

    Args:
        kps (np.ndarray): an array of shape (J, 3) for keypoints. Expect the
            last column to be the visibility in [0, 1].
        img (np.ndarray): a RGB image of shape (H, W, 3) in float32 or uint8.
        skeleton_cls (Union[str, Callable[..., Skeleton]]): a class name or a
            callable that returns a Skeleton instance.
        rgbs (Optional[np.ndarray]): A set of rgbs for each keypoint of shape
            (J, 3) or (3,). If None then use skeleton palette. Default: None.
        kp_radius (int): the radius of kps for visualization. Default: 4.
        bone_thickness (int): the thickness of bones connecting kps for
            visualization. Default: 3.

    Returns:
        combined (np.ndarray): Keypoint visualzation image of shape (H, W, 3)
            in uint8.
    """
    if isinstance(skeleton, str):
        skeleton = SKELETON_MAP[skeleton]
    if isinstance(skeleton, Callable):
        skeleton = skeleton(num_kps=len(kps), **kwargs)
    if rgbs is not None:
        if rgbs.ndim == 1:
            rgbs = rgbs[None, :].repeat(skeleton.num_kps, axis=0)
        skeleton = deepcopy(skeleton)
        skeleton._palette = rgbs.tolist()

    assert skeleton.num_kps == len(kps)

    kps = np.array(kps)
    img = image.to_uint8(img)

    H, W = img.shape[:2]
    canvas = img.copy()

    mask = (
        (kps[:, -1] != 0)
        & (kps[:, 0] >= 0)
        & (kps[:, 0] < W)
        & (kps[:, 1] >= 0)
        & (kps[:, 1] < H)
    )

    # Visualize bones.
    palette = skeleton.non_root_palette
    bones = skeleton.non_root_bones
    for rgb, (j, p) in zip(palette, bones):
        # Skip invisible keypoints.
        if (~mask[[j, p]]).any():
            continue

        kp_p, kp_j = kps[p, :2], kps[j, :2]
        kp_mid = (kp_p + kp_j) / 2
        bone_length = np.linalg.norm(kp_j - kp_p)
        bone_angle = (
            (np.arctan2(kp_j[1] - kp_p[1], kp_j[0] - kp_p[0])) * 180 / np.pi
        )
        polygon = cv2.ellipse2Poly(
            (int(kp_mid[0]), int(kp_mid[1])),
            (int(bone_length / 2), bone_thickness),
            int(bone_angle),
            arcStart=0,
            arcEnd=360,
            delta=5,
        )
        cv2.fillConvexPoly(canvas, polygon, rgb, lineType=cv2.LINE_AA)
    canvas = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

    # Visualize keypoints.
    combined = canvas.copy()
    palette = skeleton.palette
    for rgb, kp, valid in zip(palette, kps, mask):
        # Skip invisible keypoints.
        if not valid:
            continue

        cv2.circle(
            combined,
            (int(kp[0]), int(kp[1])),
            radius=kp_radius,
            color=rgb,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    combined = cv2.addWeighted(canvas, 0.3, combined, 0.7, 0)

    return combined
