#!/usr/bin/env python3
#
# File   : bkgd.py
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

import gin
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from jax import random

from dycheck.utils import struct, types

from . import utils


@gin.configurable(allowlist=["noise_std"])
def compute_bkgd_loss(
    model: nn.Module,
    key: types.PRNGKey,
    variables: FrozenDict,
    bkgd_points: jnp.ndarray,
    noise_std: float = 0.001,
    alpha: float = -2,
    scale: float = 0.001,
) -> jnp.ndarray:
    key0, key1 = random.split(key)

    bkgd_points = bkgd_points + noise_std * random.normal(
        key0, bkgd_points.shape
    )
    samples = struct.Samples(
        xs=bkgd_points,
        directions=jnp.zeros_like(bkgd_points),
        metadata=struct.Metadata(
            time=random.randint(
                key1,
                (bkgd_points.shape[0], 1),
                minval=0,
                maxval=model.num_points_embeds,
                dtype=jnp.uint32,
            )
        ),
    )
    out = model.apply(
        variables,
        samples,
        extra_params=None,
        method=lambda module, *args, **kwargs: module.points_embed.warp_v2c(
            *args, **kwargs
        ),
    )
    warped_points = out["warped_points"]
    sq_residual = jnp.sum((warped_points - bkgd_points) ** 2, axis=-1)
    loss = utils.general_loss_with_squared_residual(
        sq_residual, alpha=alpha, scale=scale
    )
    loss = loss.mean()
    return loss
