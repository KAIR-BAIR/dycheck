#!/usr/bin/env python3
#
# File   : common.py
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

import jax.numpy as jnp
import numpy as np

from dycheck.utils import types


def masked_mean(
    x: types.Array, mask: Optional[types.Array] = None
) -> types.Array:
    """Compute mean of masked values by soft blending.

    Support both jnp.ndarray and np.ndarray.

    Args:
        x (types.Array): Input array of shape (...,).
        mask (types.Array): Mask array in [0, 1]. Shape will be broadcasted to
            match x.

    Returns:
        types.Array: Masked mean of x of shape ().
    """
    eps = 1e-6

    broadcast_to = (
        jnp.broadcast_to if isinstance(x, jnp.ndarray) else np.broadcast_to
    )
    if mask is None:
        return x.mean()

    mask = broadcast_to(mask, x.shape)
    return (x * mask).sum() / mask.sum().clip(eps)  # type: ignore
