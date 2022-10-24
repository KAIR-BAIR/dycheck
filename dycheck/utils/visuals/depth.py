#!/usr/bin/env python3
#
# File   : depth.py
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

from typing import Callable, Optional, Union

import numpy as np
from matplotlib import cm

from dycheck.utils import image


def visualize_depth(
    depth: np.ndarray,
    acc: Optional[np.ndarray] = None,
    near: Optional[float] = None,
    far: Optional[float] = None,
    ignore_frac: float = 0,
    curve_fn: Callable = lambda x: -np.log(x + np.finfo(np.float32).eps),
    cmap: Union[str, Callable] = "turbo",
    invalid_depth: float = 0,
) -> np.ndarray:
    """Visualize a depth map.

    Args:
        depth (np.ndarray): A depth map of shape (H, W, 1).
        acc (np.ndarray): An accumulation map of shape (H, W, 1) in [0, 1].
        near (Optional[float]): The depth of the near plane. If None then just
            use the min. Default: None.
        far (Optional[float]): The depth of the far plane. If None then just
            use the max. Default: None.
        ignore_frac (float): The fraction of the depth map to ignore when
            automatically generating `near` and `far`. Depends on `acc` as well
            as `depth'. Default: 0.
        curve_fn (Callable): A curve function that gets applied to `depth`,
            `near`, and `far` before the rest of visualization. Good choices:
            x, 1/(x+eps), log(x+eps). Note that the default choice will flip
            the sign of depths, so that the default cmap (turbo) renders "near"
            as red and "far" as blue. Default: a negative log scale mapping.
        cmap (Union[str, Callable]): A cmap for colorization. Default: "turbo".
        invalid_depth (float): The value to use for invalid depths. Can be
            np.nan. Default: 0.

    Returns:
        np.ndarray: A depth visualzation image of shape (H, W, 3) in uint8.
    """
    depth = np.array(depth)
    if acc is None:
        acc = np.ones_like(depth)
    else:
        acc = np.array(acc)
    if invalid_depth is not None:
        if invalid_depth is np.nan:
            acc = np.where(np.isnan(depth), np.zeros_like(acc), acc)
        else:
            acc = np.where(depth == invalid_depth, np.zeros_like(acc), acc)

    if near is None or far is None:
        # Sort `depth` and `acc` according to `depth`, then identify the depth
        # values that span the middle of `acc`, ignoring `ignore_frac` fraction
        # of `acc`.
        sortidx = np.argsort(depth.reshape((-1,)))
        depth_sorted = depth.reshape((-1,))[sortidx]
        acc_sorted = acc.reshape((-1,))[sortidx]  # type: ignore
        cum_acc_sorted = np.cumsum(acc_sorted)
        mask = (cum_acc_sorted >= cum_acc_sorted[-1] * ignore_frac) & (
            cum_acc_sorted <= cum_acc_sorted[-1] * (1 - ignore_frac)
        )
        if invalid_depth is not None:
            mask &= (
                (depth_sorted != invalid_depth)
                if invalid_depth is not np.nan
                else ~np.isnan(depth_sorted)
            )
        depth_keep = depth_sorted[mask]
        eps = np.finfo(np.float32).eps
        # If `near` or `far` are None, use the highest and lowest non-NaN
        # values in `depth_keep` as automatic near/far planes.
        near = near or depth_keep[0] - eps
        far = far or depth_keep[-1] + eps

    assert near < far

    # Curve all values.
    depth, near, far = [curve_fn(x) for x in [depth, near, far]]

    # Scale to [0, 1].
    value = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1)
    )[..., 0]

    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    color = cmap(value)[..., :3]

    # Set non-accumulated pixels to white.
    color = color * acc + (1 - acc)  # type: ignore

    return image.to_uint8(color)
