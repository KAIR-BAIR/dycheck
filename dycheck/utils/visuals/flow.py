#!/usr/bin/env python3
#
# File   : flow.py
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

from typing import Optional

import cv2
import jax
import numpy as np

from .. import image
from .corrs import visualize_corrs


def _make_colorwheel() -> np.ndarray:
    """Generates a classic color wheel for optical flow visualization.

    A Database and Evaluation Methodology for Optical Flow.
        Baker et al., ICCV 2007.
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        colorwheel (np.ndarray): Color wheel of shape (55, 3) in uint8.
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def _flow_to_colors(flow: np.ndarray) -> np.ndarray:
    """Applies the flow color wheel to (possibly clipped) flow visualization
    image.

    According to the C++ source code of Daniel Scharstein.
    According to the Matlab source code of Deqing Sun.

    Args:
        flow (np.ndarray): Flow image of shape (H, W, 2).

    Returns:
        flow_visual (np.ndarray): Flow visualization image of shape (H, W, 3).
    """
    u, v = jax.tree_map(lambda x: x[..., 0], np.split(flow, 2, axis=-1))

    flow_visual = np.zeros(flow.shape[:2] + (3,), np.uint8)
    colorwheel = _make_colorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # Out of range.
        flow_visual[..., i] = np.floor(255 * col)
    return flow_visual


def visualize_flow(
    flow: np.ndarray,
    *,
    clip_flow: Optional[float] = None,
    rad_max: Optional[float] = None,
) -> np.ndarray:
    """Visualizei a flow image.

    Args:
        flow (np.ndarray): A flow image of shape (H, W, 2).
        clip_flow (Optional[float]): Clip flow to [0, clip_flow].
        rad_max (Optional[float]): Maximum radius of the flow visualization.

    Returns:
        np.ndarray: Flow visualization image of shape (H, W, 3).
    """
    flow = np.array(flow)

    if clip_flow is not None:
        flow = np.clip(flow, 0, clip_flow)
    rad = np.linalg.norm(flow, axis=-1, keepdims=True).clip(min=1e-6)
    if rad_max is None:
        rad_max = np.full_like(rad, rad.max())
    else:
        # Clip big flow to rad_max while homogenously scaling for the rest.
        rad_max = rad.clip(min=rad_max)
    flow = flow / rad_max

    return _flow_to_colors(flow)


def visualize_flow_arrows(
    flow: np.ndarray,
    img: np.ndarray,
    *,
    rgbs: Optional[np.ndarray] = None,
    clip_flow: Optional[float] = None,
    min_thresh: float = 5,
    subsample: int = 50,
    num_min_keeps: int = 10,
    line_thickness: int = 1,
    tip_length: float = 0.2,
    alpha: float = 0.5,
) -> np.ndarray:
    """Visualize a flow image with arrows.

    Args:
        flow (np.ndarray): A flow image of shape (H, W, 2).
        img (np.ndarray): An image for start points of shape (H, W, 3) in
            float32 or uint8.
        rgbs (Optional[np.ndarray]): A color map for the arrows at each pixel
            location of shape (H, W, 3). Default: None.
        clip_flow (Optional[float]): Clip flow to [0, clip_flow].
        min_thresh (float): Minimum threshold for flow magnitude.
        subsample (int): Subsample the flow to speed up visualization.
        num_min_keeps (int): The number of correspondences to keep. Default:
            10.
        line_thickness (int): Line thickness. Default: 1.
        tip_length (float): Length of the arrow tip. Default: 0.2.
        alpha (float): The alpha value between [0, 1] for foreground blending.
            The bigger the more prominent of the visualization. Default: 0.5.

    Returns:
        canvas (np.ndarray): Flow visualization image of shape (H, W, 3).
    """
    img = image.to_uint8(img)
    canvas = img.copy()
    rng = np.random.default_rng(0)

    if rgbs is None:
        rgbs = visualize_flow(flow, clip_flow=clip_flow)
    H, W = flow.shape[:2]

    flow_start = np.stack(np.meshgrid(range(W), range(H)), 2)
    flow_end = (
        flow[flow_start[..., 1], flow_start[..., 0]] + flow_start
    ).astype(np.int32)

    norm = np.linalg.norm(flow, axis=-1)
    valid_mask = (
        (norm >= min_thresh)
        & (flow_end[..., 0] < flow.shape[1])
        & (flow_end[..., 0] >= 0)
        & (flow_end[..., 1] < flow.shape[0])
        & (flow_end[..., 1] >= 0)
    )
    filtered_inds = np.stack(np.nonzero(valid_mask), axis=-1)
    num_min_keeps = min(
        max(num_min_keeps, filtered_inds.shape[0] // subsample),
        filtered_inds.shape[0],
    )
    filtered_inds = (
        rng.choice(filtered_inds, num_min_keeps, replace=False)
        if filtered_inds.shape[0] > 0
        else []
    )

    for inds in filtered_inds:
        y, x = inds
        start = tuple(flow_start[y, x])
        end = tuple(flow_end[y, x])
        rgb = tuple(int(x) for x in rgbs[y, x])
        cv2.arrowedLine(
            canvas,
            start,
            end,
            color=rgb,
            thickness=line_thickness,
            tipLength=tip_length,
            line_type=cv2.LINE_AA,
        )

    canvas = cv2.addWeighted(img, alpha, canvas, 1 - alpha, 0)
    return canvas


def visualize_flow_corrs(
    flow: np.ndarray,
    img: np.ndarray,
    img_to: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    rgbs: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Visualize a flow image as a set of correspondences.

    Args:
        flow (np.ndarray): A flow image of shape (H, W, 2).
        img (np.ndarray): An image for start points of shape (H, W, 3) in
            float32 or uint8.
        img_to (np.ndarray): An image for end points of shape (H, W, 3) in
            float32 or uint8.
        mask (Optional[np.ndarray]): A hard mask for start points of shape
            (H, W, 1). Default: None.
        rgbs (Optional[np.ndarray]): A color map for the arrows at each pixel
            location of shape (H, W, 3). Default: None.

    Returns:
        canvas (np.ndarray): Flow visualization image of shape (H, W, 3).
    """
    flow_start = np.stack(
        np.meshgrid(range(flow.shape[1]), range(flow.shape[0])), 2
    )
    flow_end = (
        flow[flow_start[..., 1], flow_start[..., 0]] + flow_start
    ).astype(np.int32)
    flow_corrs = np.stack([flow_start, flow_end])

    if mask is not None:
        # Only show correspondences inside of the mask.
        flow_corrs = flow_corrs[np.stack([mask[..., 0]] * 2)]
        if rgbs is not None:
            rgbs = rgbs[mask[..., 0]]
        if flow_corrs.shape[0] == 0:
            flow_corrs = np.ones((0, 4))
            if rgbs is not None:
                rgbs = np.ones((0, 3))
    flow_corrs = flow_corrs.reshape(2, -1, 2).swapaxes(0, 1)

    return visualize_corrs(flow_corrs, img, img_to, rgbs=rgbs, **kwargs)
