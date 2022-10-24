#!/usr/bin/env python3
#
# File   : emf.py
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

import functools
from typing import Optional, Sequence

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

from dycheck import geometry
from dycheck.utils import common

from . import get_compute_dpt_disp, get_compute_raft_flow


def compute_angular_emf(
    orientations: np.ndarray,
    positions: np.ndarray,
    fps: float,
    *,
    lookat: Optional[np.ndarray] = None,
) -> float:
    """Compute the angular effective multi-view factor (omega) given an ordered
    sequence of camera poses and the frame rate.

    Args:
        orientations (np.ndarray): The orientation of the camera of shape (N,
            3, 3) that maps the world coordinates to camera coordinates in the
            OpenCV format.
        positions (np.ndarray): The position of the camera of shape (N, 3) in
            the world coordinates.
        fps (float): The frame rate.
        lookat (Optional[np.ndarray]): The lookat point. If None, the lookat
            point is computed by triangulating the camera optical axes.

    Returns:
        float: The angular effective multi-view factor.
    """
    if lookat is None:
        # The z-axis in the local camera space.
        optical_axes = orientations[:, 2]
        lookat = geometry.utils.tringulate_rays(positions, optical_axes)
    viewdirs = lookat - positions
    viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    return (
        np.arccos(
            (viewdirs[:-1] * viewdirs[1:]).sum(axis=-1).clip(-1, 1),
        ).mean()
        * 180
        / np.pi
        * fps
    ).item()


def _chunk_remap(
    data: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    chunk: int = 16384,
    **kwargs,
) -> np.ndarray:
    _map_x = map_x.reshape(-1)
    _map_y = map_y.reshape(-1)
    outs = []
    for i in range(0, _map_x.shape[0], chunk):
        out = cv2.remap(
            data,
            _map_x[i : i + chunk],
            _map_y[i : i + chunk],
            **kwargs,
        )
        outs.append(out)
    outs = np.concatenate(outs, axis=0).reshape(
        map_x.shape + (data.shape[-1],)
        if len(data.shape) == 3
        else map_x.shape
    )
    return outs


def _solve_points(
    camera: geometry.Camera,
    pred_disp: np.ndarray,
    *,
    bkgd_points: np.ndarray,
) -> np.ndarray:
    pixels, pixel_depths = camera.project(bkgd_points, return_depth=True)
    valid = pixel_depths[..., 0] > 0
    pixels = pixels[valid]
    pixel_depths = pixel_depths[valid]
    pixel_disps = 1 / pixel_depths

    pred_pixel_disps = _chunk_remap(
        pred_disp,
        np.array(pixels[..., 0]),
        np.array(pixels[..., 1]),
        interpolation=cv2.INTER_LINEAR,
    )

    H, W = camera.image_shape
    valid = (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] < W)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < H)
        & (pixel_disps[..., 0] > 0)
        & (pred_pixel_disps[..., 0] > 0)
    )
    X = pred_pixel_disps[valid]
    Y = pixel_disps[valid]
    regressor = RANSACRegressor(random_state=0).fit(X, Y)

    pixels = camera.get_pixels()
    transformed_disp = regressor.predict(pred_disp.reshape(-1, 1)).reshape(
        H, W, 1
    )
    transformed_depth = 1 / transformed_disp
    points = camera.pixels_to_points(
        pixels=pixels,
        depth=transformed_depth,
    )

    return points


def _solve_ratio(
    flow: np.ndarray,
    occ: np.ndarray,
    point_map: np.ndarray,
    point_map_to: np.ndarray,
    camera: geometry.Camera,
    camera_to: geometry.Camera,
) -> float:
    if (occ < 0.5).sum() == 0:
        return None

    camera_delta = np.linalg.norm(camera_to.position - camera.position)
    if camera_delta == 0:
        return None

    pixels = camera.get_pixels()
    pixels_to = pixels + flow

    H, W = camera.image_shape
    valid = (
        (pixels_to[..., 0] >= 0)
        & (pixels_to[..., 0] < W)
        & (pixels_to[..., 1] >= 0)
        & (pixels_to[..., 1] < H)
    )
    points_to = _chunk_remap(
        point_map_to,
        pixels_to[valid, 0],
        pixels_to[valid, 1],
        interpolation=cv2.INTER_LINEAR,
    )
    point_deltas = np.linalg.norm(points_to - point_map[valid], axis=-1)[
        occ[valid, 0] < 0.5
    ]
    point_delta = point_deltas[
        point_deltas < np.quantile(point_deltas, 0.95)
    ].mean()

    ratio = (camera_delta / point_delta).item()
    return ratio


def compute_full_emf(
    rgbs: np.ndarray,
    cameras: Sequence[geometry.Camera],
    bkgd_points: np.ndarray,
) -> float:
    """Compute the full effective multi-view factor (Omega) given an ordered
    sequence of rgb images, corresponding cameras and the anchor background
    points to solve for relative scale .

    Args:
        rgbs (np.ndarray): An array of shape (N, H, W, 3) representing the
            video frames, in either uint8 or float32.
        cameras (Sequence[geometry.Camera]): A sequence of camera objects of
            corresponding frames.
        bkgd_points (np.ndarray): An array of shape (P, 3) for the anchor
            background points to solve for the relative scale.

    Returns:
        float: The full effective multi-view factor.
    """
    compute_raft_flow = get_compute_raft_flow(
        chunk=16, desc="* Compute RAFT flow"
    )
    raft_flows = compute_raft_flow(rgbs[:-1], rgbs[1:])

    compute_dpt_disp = get_compute_dpt_disp()
    dpt_disps = np.array(
        [
            compute_dpt_disp(rgb)
            for rgb in common.tqdm(rgbs, desc="* Compute DPT depth")
        ]
    ).astype(np.float32)

    bkgd_points = bkgd_points.astype(np.float32)

    points = common.parallel_map(
        functools.partial(_solve_points, bkgd_points=bkgd_points),
        cameras,
        dpt_disps,
        show_pbar=True,
        desc="* Solve points",
    )

    ratios_fw = common.parallel_map(
        _solve_ratio,
        raft_flows.flow_fw,
        raft_flows.occ_fw,
        points[:-1],
        points[1:],
        cameras[:-1],
        cameras[1:],
        show_pbar=True,
        desc="* Solve ratios",
    )
    ratios_bw = common.parallel_map(
        _solve_ratio,
        raft_flows.flow_bw,
        raft_flows.occ_bw,
        points[1:],
        points[:-1],
        cameras[1:],
        cameras[:-1],
    )
    return np.mean(
        [r for r in ratios_fw if r is not None]
        + [r for r in ratios_bw if r is not None]
    ).item()
