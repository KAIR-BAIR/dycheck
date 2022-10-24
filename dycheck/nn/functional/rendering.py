#!/usr/bin/env python3
#
# File   : rendering.py
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

from typing import Dict, Optional

import jax.numpy as jnp
from jax import random

from dycheck.utils import struct, types


def perturb_logits(
    key: types.PRNGKey,
    logits: Dict[str, jnp.ndarray],
    use_randomized: bool,
    noise_std: Optional[float],
) -> Dict[str, jnp.ndarray]:
    """Regularize the sigma prediction by adding gaussian noise.

    Args:
        key (types.PRNGKey): A random number generator.
        logits (Dict[str, Any]): A dictionary holding at least "sigma".
        use_randomized (Dict[str, jnp.ndarray]): Add noise only if
            use_randomized is True and noise_std is bigger than 0,
        noise_std (Optional[float]): Standard dev of noise added to regularize
            sigma output.

    Returns:
        logits: Updated
            logits.
    """
    if use_randomized and noise_std is not None and noise_std > 0.0:
        assert "point_sigma" in logits
        key = random.split(key)[1]
        noise = (
            random.normal(key, logits["point_sigma"].shape, dtype=logits.dtype)
            * noise_std
        )
        logits["point_sigma"] += noise
    return logits


def volrend(
    out: Dict[str, jnp.ndarray],
    samples: struct.Samples,
    bkgd_rgb: jnp.ndarray,
    use_sample_at_infinity: bool,
    eps: float = 1e-10,
) -> Dict[str, jnp.ndarray]:
    """Render through volume by numerical integration.

    Args:
        out (Dict[str, jnp.ndarray]): A dictionary holding at least
            "point_sigma" (..., S, 1) and "point_rgb" (..., S, 3).
        samples (struct.Samples): Samples to render of shape (..., S).
        bkgd_rgb (jnp.ndarray): Background color of shape (3,).
        use_sample_at_infinity (bool): Whether to sample at infinity.

    Returns:
        Dict[str, jnp.ndarray]: rendering results.
    """
    assert samples.tvals is not None
    assert "point_sigma" in out and "point_rgb" in out
    batch_shape = samples.xs.shape[:-1]

    # TODO(keunhong): Remove this hack.
    # NOTE(Hang Gao @ 07/15): Actually needed by Nerfies & HyperNeRF to always
    # stop by the scene bound otherwise it will not when trained without depth.
    last_sample_t = 1e10 if use_sample_at_infinity else 1e-19

    # (..., S, 1)
    dists = jnp.concatenate(
        [
            samples.tvals[..., 1:, :] - samples.tvals[..., :-1, :],
            jnp.broadcast_to([last_sample_t], batch_shape[:-1] + (1, 1)),
        ],
        -2,
    )
    dists = dists * jnp.linalg.norm(samples.directions, axis=-1, keepdims=True)

    # (..., S, 1)
    alpha = 1 - jnp.exp(-out["point_sigma"] * dists)
    # Prepend a 1 to make this an 'exclusive' cumprod as in `tf.math.cumprod`.
    # (..., S, 1)
    trans = jnp.concatenate(
        [
            jnp.ones_like(alpha[..., :1, :], alpha.dtype),
            jnp.cumprod(1 - alpha[..., :-1, :] + eps, axis=-2),
        ],
        axis=-2,
    )
    # (..., S, 1)
    weights = alpha * trans

    # (..., 1)
    acc = (
        weights[..., :-1, :].sum(axis=-2)
        if use_sample_at_infinity
        else weights.sum(axis=-2)
    )

    # (..., 1)
    # Avoid 0/0 case.
    depth = (
        weights[..., :-1, :] * samples.tvals[..., :-1, :]
        if use_sample_at_infinity
        else weights * samples.tvals
    ).sum(axis=-2) / acc.clip(1e-12)
    # This nan_to_num trick from Jon does not really work for the 0/0 case and
    # will cause NaN gradient.
    depth = jnp.clip(
        jnp.nan_to_num(depth, nan=jnp.inf),
        samples.tvals[..., 0, :],
        samples.tvals[..., -1, :],
    )

    # (..., 3)
    rgb = (weights * out["point_rgb"]).sum(axis=-2)
    rgb = rgb + bkgd_rgb * (1 - acc)

    out = {
        "alpha": alpha,
        "trans": trans,
        "weights": weights,
        "acc": acc,
        "depth": depth,
        "rgb": rgb,
    }
    return out
