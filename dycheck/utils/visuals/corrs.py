#!/usr/bin/env python3
#
# File   : corrs.py
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
import numpy as np

from .. import image


def visualize_corrs(
    corrs: np.ndarray,
    img: np.ndarray,
    img_to: np.ndarray,
    *,
    rgbs: Optional[np.ndarray] = None,
    min_rad: float = 5,
    subsample: int = 50,
    num_min_keeps: int = 10,
    circle_radius: int = 1,
    circle_thickness: int = -1,
    line_thickness: int = 1,
    alpha: float = 0.7,
):
    """Visualize a set of correspondences.

    By default this function visualizes a sparse set subset of correspondences
    with lines.

    Args:
        corrs (np.ndarray): A set of correspondences of shape (N, 2, 2), where
            the second dimension represents (from, to) and the last dimension
            represents (x, y) coordinates.
        img (np.ndarray): An image for start points of shape (Hi, Wi, 3) in
            either float32 or uint8.
        img_to (np.ndarray): An image for end points of shape (Ht, Wt, 3) in
            either float32 or uint8.
        rgbs (Optional[np.ndarray]): A set of rgbs for each correspondence
            of shape (N, 3) or (3,). If None then use pixel coordinates.
            Default: None.
        min_rad (float): The minimum threshold for the correspondence.
        subsample (int): The number of points to subsample. Default: 50.
        num_min_keeps (int): The number of correspondences to keep. Default:
            10.
        circle_radius (int): The radius of the circle. Default: 1.
        circle_thickness (int): The thickness of the circle. Default: 1.
        line_thickness (int): The thickness of the line. Default: 1.
        alpha (float): The alpha value between [0, 1] for foreground blending.
            The bigger the more prominent of the visualization. Default: 0.7.

    Returns:
        np.ndarray: A visualization image of shape (H, W, 3) in uint8.
    """
    corrs = np.array(corrs)
    img = image.to_uint8(img)
    img_to = image.to_uint8(img_to)
    rng = np.random.default_rng(0)

    (Hi, Wi), (Ht, Wt) = img.shape[:2], img_to.shape[:2]
    combined = np.concatenate([img, img_to], axis=1)
    canvas = combined.copy()

    norm = np.linalg.norm(corrs[:, 1] - corrs[:, 0], axis=-1)
    mask = (
        (norm >= min_rad)
        & (corrs[..., 0, 0] < Wi)
        & (corrs[..., 0, 0] >= 0)
        & (corrs[..., 0, 1] < Hi)
        & (corrs[..., 0, 1] >= 0)
        & (corrs[..., 1, 0] < Wt)
        & (corrs[..., 1, 0] >= 0)
        & (corrs[..., 1, 1] < Ht)
        & (corrs[..., 1, 1] >= 0)
    )
    filtered_inds = np.nonzero(mask)[0]
    num_min_keeps = min(
        max(num_min_keeps, filtered_inds.shape[0] // subsample),
        filtered_inds.shape[0],
    )
    filtered_inds = (
        rng.choice(filtered_inds, num_min_keeps, replace=False)
        if filtered_inds.shape[0] > 0
        else []
    )

    if len(filtered_inds) > 0:
        if rgbs is None:
            # Use normalized pixel coordinate of img for colorization.
            corr = corrs[:, 0]
            phi = 2 * np.pi * (corr[:, 0] / (Wi - 1) - 0.5)
            theta = np.pi * (corr[:, 1] / (Hi - 1) - 0.5)
            x = np.cos(theta) * np.cos(phi)
            y = np.cos(theta) * np.sin(phi)
            z = np.sin(theta)
            rgbs = image.to_uint8((np.stack([x, y, z], axis=-1) + 1) / 2)
        for idx in filtered_inds:
            start = tuple(corrs[idx, 0].astype(np.int32))
            end = tuple((corrs[idx, 1] + [Wi, 0]).astype(np.int32))
            rgb = tuple(
                int(c) for c in (rgbs[idx] if rgbs.ndim == 2 else rgbs)
            )
            cv2.circle(
                combined,
                start,
                radius=circle_radius,
                color=rgb,
                thickness=circle_thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                combined,
                end,
                radius=circle_radius,
                color=rgb,
                thickness=circle_thickness,
                lineType=cv2.LINE_AA,
            )
            if line_thickness > 0:
                cv2.line(
                    canvas,
                    start,
                    end,
                    color=rgb,
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )

    combined = cv2.addWeighted(combined, alpha, canvas, 1 - alpha, 0)
    return combined


def visualize_chained_corrs(
    corrs: np.ndarray,
    imgs: np.ndarray,
    *,
    rgbs: Optional[np.ndarray] = None,
    circle_radius: int = 1,
    circle_thickness: int = -1,
    line_thickness: int = 1,
    alpha: float = 0.7,
):
    """Visualize a set of correspondences.

    By default this function visualizes a sparse set subset of correspondences
    with lines.

    Args:
        corrs (np.ndarray): A set of correspondences of shape (N, C, 2), where
            the second dimension represents chained frames and the last
            dimension represents (x, y) coordinates.
        imgs (np.ndarray): An image for start points of shape (C, H, W, 3) in
            either float32 or uint8.
        rgbs (Optional[np.ndarray]): A set of rgbs for each correspondence
            of shape (N, 3) or (3,). If None then use pixel coordinates.
            Default: None.
        circle_radius (int): The radius of the circle. Default: 1.
        circle_thickness (int): The thickness of the circle. Default: 1.
        line_thickness (int): The thickness of the line. Default: 1.
        alpha (float): The alpha value between [0, 1] for foreground blending.
            The bigger the more prominent of the visualization. Default: 0.7.

    Returns:
        np.ndarray: A visualization image of shape (H, W, 3) in uint8.
    """
    corrs = np.array(corrs)
    imgs = image.to_uint8(imgs)

    C, H, W = imgs.shape[:3]
    combined = np.concatenate(list(imgs), axis=1)
    canvas = combined.copy()

    if rgbs is None:
        # Use normalized pixel coordinate of img for colorization.
        corr = corrs[:, 0]
        phi = 2 * np.pi * (corr[:, 0] / (W - 1) - 0.5)
        theta = np.pi * (corr[:, 1] / (H - 1) - 0.5)
        x = np.cos(theta) * np.cos(phi)
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)
        rgbs = image.to_uint8((np.stack([x, y, z], axis=-1) + 1) / 2)

    for i in range(C):
        mask = (
            (corrs[..., i, 0] < W)
            & (corrs[..., i, 0] >= 0)
            & (corrs[..., i, 1] < H)
            & (corrs[..., i, 1] >= 0)
        )
        filtered_inds = np.nonzero(mask)[0]

        for idx in filtered_inds:
            start = tuple((corrs[idx, i] + [W * i, 0]).astype(np.int32))
            rgb = tuple(
                int(c) for c in (rgbs[idx] if rgbs.ndim == 2 else rgbs)
            )
            cv2.circle(
                combined,
                start,
                radius=circle_radius,
                color=rgb,
                thickness=circle_thickness,
                lineType=cv2.LINE_AA,
            )
            if (
                line_thickness > 0
                and i < C - 1
                and (
                    (corrs[idx, i + 1, 0] < W)
                    & (corrs[idx, i + 1, 0] >= 0)
                    & (corrs[idx, i + 1, 1] < H)
                    & (corrs[idx, i + 1, 1] >= 0)
                )
            ):
                end = tuple(
                    (corrs[idx, i + 1] + [W * (i + 1), 0]).astype(np.int32)
                )
                cv2.line(
                    canvas,
                    start,
                    end,
                    color=rgb,
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )

    combined = cv2.addWeighted(combined, alpha, canvas, 1 - alpha, 0)
    return combined
