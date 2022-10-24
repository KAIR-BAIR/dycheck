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

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.representation.ray import (
    triangulate as ray_triangulate,
)

from dycheck.utils import types


def matmul(a: types.Array, b: types.Array) -> types.Array:
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
    else:
        assert isinstance(a, jnp.ndarray)
        assert isinstance(b, jnp.ndarray)

    if isinstance(a, np.ndarray):
        return a @ b
    else:
        # NOTE: The original implementation uses highest precision for TPU
        # computation. Since we are using GPUs only, comment it out.
        #  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)
        return jnp.matmul(a, b)


def matv(a: types.Array, b: types.Array) -> types.Array:
    return matmul(a, b[..., None])[..., 0]


def tringulate_rays(origins: types.Array, viewdirs: types.Array) -> np.ndarray:
    """Triangulate a set of rays to find a single lookat point.

    Args:
        origins (types.Array): A (N, 3) array of ray origins.
        viewdirs (types.Array): A (N, 3) array of ray view directions.

    Returns:
        np.ndarray: A (3,) lookat point.
    """
    tf.config.set_visible_devices([], "GPU")

    origins = np.array(origins[None], np.float32)
    viewdirs = np.array(viewdirs[None], np.float32)
    weights = np.ones(origins.shape[:2], dtype=np.float32)
    points = np.array(ray_triangulate(origins, origins + viewdirs, weights))
    return points[0]
