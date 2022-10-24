#!/usr/bin/env python3
#
# File   : dist.py
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

import jax
import jax.numpy as jnp


@jax.jit
def compute_dist_loss(
    pred_weights: jnp.ndarray, svals: jnp.ndarray
) -> jnp.ndarray:
    """Compute the distortion loss of each ray.

    Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields.
        Barron et al., CVPR 2022.
        https://arxiv.org/abs/2111.12077

    As per Equation (15) in the paper. Note that we slightly modify the loss to
    account for "sampling at infinity" when rendering NeRF.

    Args:
        pred_weights (jnp.ndarray): (..., S, 1) predicted weights of each
            sample along the ray.
        svals (jnp.ndarray): (..., S + 1, 1) normalized marching step of each
            sample along the ray.
    """
    pred_weights = pred_weights[..., 0]

    # (..., S)
    smids = 0.5 * (svals[..., 1:, 0] + svals[..., :-1, 0])
    sdeltas = svals[..., 1:, 0] - svals[..., :-1, 0]

    loss1 = (
        pred_weights[..., None, :]
        * pred_weights[..., None]
        * jnp.abs(smids[..., None, :] - smids[..., None])
    ).sum(axis=(-2, -1))
    # (...)
    loss2 = 1 / 3 * (pred_weights**2 * sdeltas).sum(axis=-1)
    loss = loss1 + loss2
    return loss.mean()
