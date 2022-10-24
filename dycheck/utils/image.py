#!/usr/bin/env python3
#
# File   : image.py
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

import math
from typing import Any, Tuple

import cv2
import jax.numpy as jnp
import numpy as np
from absl import logging

from . import types

UINT8_MAX = 255
UINT16_MAX = 65535


def downscale(img: types.Array, scale: int) -> np.ndarray:
    if isinstance(img, jnp.ndarray):
        img = np.array(img)

    if scale == 1:
        return img

    height, width = img.shape[:2]
    if height % scale > 0 or width % scale > 0:
        raise ValueError(
            f"Image shape ({height},{width}) must be divisible by the"
            f" scale ({scale})."
        )
    out_height, out_width = height // scale, width // scale
    resized = cv2.resize(img, (out_width, out_height), cv2.INTER_AREA)
    return resized


def upscale(img: types.Array, scale: int) -> np.ndarray:
    if isinstance(img, jnp.ndarray):
        img = np.array(img)

    if scale == 1:
        return img

    height, width = img.shape[:2]
    out_height, out_width = height * scale, width * scale
    resized = cv2.resize(img, (out_width, out_height), cv2.INTER_AREA)
    return resized


def rescale(
    img: types.Array, scale_factor: float, interpolation: Any = cv2.INTER_AREA
) -> np.ndarray:
    scale_factor = float(scale_factor)

    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be a non-negative number.")
    if scale_factor == 1.0:
        return img

    height, width = img.shape[:2]
    if scale_factor.is_integer():
        return upscale(img, int(scale_factor))

    inv_scale = 1.0 / scale_factor
    if (
        inv_scale.is_integer()
        and (scale_factor * height).is_integer()
        and (scale_factor * width).is_integer()
    ):
        return downscale(img, int(inv_scale))

    logging.warning(
        "Resizing image by non-integer factor %f, this may lead to artifacts.",
        scale_factor,
    )

    height, width = img.shape[:2]
    out_height = math.ceil(height * scale_factor)
    out_height -= out_height % 2
    out_width = math.ceil(width * scale_factor)
    out_width -= out_width % 2

    return resize(img, (out_height, out_width), interpolation)


def resize(
    img: types.Array,
    shape: Tuple[int, int],
    interpolation: Any = cv2.INTER_AREA,
) -> np.ndarray:
    if isinstance(img, jnp.ndarray):
        img = np.array(img)

    out_height, out_width = shape
    return cv2.resize(
        img,
        (out_width, out_height),
        interpolation=interpolation,
    )


def varlap(img: types.Array) -> np.ndarray:
    """Measure the focus/motion-blur of an image by the Laplacian variance."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def to_float32(img: types.Array) -> np.ndarray:
    img = np.array(img)
    if img.dtype == np.float32:
        return img

    dtype = img.dtype
    img = img.astype(np.float32)
    if dtype == np.uint8:
        return img / UINT8_MAX
    elif dtype == np.uint16:
        return img / UINT16_MAX
    elif dtype == np.float64:
        return img
    elif dtype == np.float16:
        return img

    raise ValueError(f"Unexpected dtype: {dtype}.")


def to_quantized_float32(img: types.Array) -> np.ndarray:
    return to_float32(to_uint8(img))


def to_uint8(img: types.Array) -> np.ndarray:
    img = np.array(img)
    if img.dtype == np.uint8:
        return img
    if not issubclass(img.dtype.type, np.floating):
        raise ValueError(
            f"Input image should be a floating type but is of type "
            f"{img.dtype!r}."
        )
    return (img * UINT8_MAX).clip(0.0, UINT8_MAX).astype(np.uint8)


def to_uint16(img: types.Array) -> np.ndarray:
    img = np.array(img)
    if img.dtype == np.uint16:
        return img
    if not issubclass(img.dtype.type, np.floating):
        raise ValueError(
            f"Input image should be a floating type but is of type "
            f"{img.dtype!r}."
        )
    return (img * UINT16_MAX).clip(0.0, UINT16_MAX).astype(np.uint16)


# Special forms of images.
def rescale_flow(
    flow: types.Array,
    scale_factor: float,
    interpolation: Any = cv2.INTER_LINEAR,
) -> np.ndarray:
    height, width = flow.shape[:2]

    out_flow = rescale(flow, scale_factor, interpolation)

    out_height, out_width = out_flow.shape[:2]
    out_flow[..., 0] *= float(out_width) / float(width)
    out_flow[..., 1] *= float(out_height) / float(height)
    return out_flow
