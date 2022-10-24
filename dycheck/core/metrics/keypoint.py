#!/usr/bin/env python3
#
# File   : keypoint.py
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

from typing import Literal, Optional, Tuple

import jax.numpy as jnp


def compute_pck(
    kps0: jnp.ndarray,
    kps1: jnp.ndarray,
    img_wh: Tuple[int, int],
    ratio: float = 0.05,
    reduce: Optional[Literal["mean"]] = "mean",
) -> jnp.ndarray:
    """Compute PCK between two sets of keypoints given the threshold ratio.

    Canonical Surface Mapping via Geometric Cycle Consistency.
        Kulkarni et al., ICCV 2019.
        https://arxiv.org/abs/1907.10043

    Args:
        kps0 (jnp.ndarray): A set of keypoints of shape (J, 2) in float32.
        kps1 (jnp.ndarray): A set of keypoints of shape (J, 2) in float32.
        img_wh (Tuple[int, int]): Image width and height.
        ratio (float): A threshold ratios. Default: 0.05.
        reduce (Optional[Literal["mean"]]): Reduction method. Default: "mean".

    Returns:
        jnp.ndarray:
            if reduce == "mean", PCK of shape();
            if reduce is None, corrects of shape (J,).
    """
    dists = jnp.linalg.norm(kps0 - kps1, axis=-1)
    thres = ratio * max(img_wh)
    corrects = dists < thres
    if reduce == "mean":
        return corrects.mean()
    elif reduce is None:
        return corrects
