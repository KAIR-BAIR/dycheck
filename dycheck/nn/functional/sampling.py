#!/usr/bin/env python3
#
# File   : sampling.py
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
from jax import lax, random

from dycheck.utils import struct, types


def _piecewise_constant_pdf(
    key: types.PRNGKey,
    bins: jnp.ndarray,
    weights: jnp.ndarray,
    num_samples: int,
    use_randomized: bool,
) -> jnp.ndarray:
    """Piecewise-constant PDF sampling from sorted bins.

    Args:
        key (types.PRNGKey): A random number generator.
        bins (jnp.ndarray): Bins of shape (batch_size, n_bins + 1).
        weights (jnp.ndarray): Weights of shape (batch_size, n_bins).
        num_samples (int): Number of samples to be generated.
        use_randomized (bool): Use stratified sampling.

    Returns:
        tvals (jnp.ndarray(float32)): (batch_size, num_samples).
    """
    eps = 1e-5

    # Get pdf
    weights += eps  # prevent nans
    pdf = weights / weights.sum(axis=-1, keepdims=True)
    cdf = jnp.cumsum(pdf, axis=-1)
    cdf = jnp.concatenate(
        [jnp.zeros(list(cdf.shape[:-1]) + [1]), cdf], axis=-1
    )

    # Take uniform samples
    if use_randomized:
        u = random.uniform(key, list(cdf.shape[:-1]) + [num_samples])
    else:
        u = jnp.linspace(0.0, 1.0, num_samples)
        u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    # Invert CDF. This takes advantage of the fact that `bins` is sorted.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # TODO(Hang Gao @ 07/15): use cumsort to speed it up.
        x0 = jnp.max(jnp.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = jnp.min(jnp.where(~mask, x[..., None], x[..., -1:, None]), -2)
        x0 = jnp.minimum(x0, x[..., -2:-1])
        x1 = jnp.maximum(x1, x[..., 1:2])
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    denom = cdf_g1 - cdf_g0
    denom = jnp.where(denom < eps, 1.0, denom)
    t = (u - cdf_g0) / denom
    tvals = bins_g0 + t * (bins_g1 - bins_g0)

    # Prevent gradient from backprop-ing through samples
    return lax.stop_gradient(tvals)


def uniform(
    key: types.PRNGKey,
    rays: struct.Rays,
    num_samples: int,
    near: float,
    far: float,
    use_randomized: bool,
    use_linear_disparity: bool,
) -> struct.Samples:
    """Uniformly sample along the ray.

    Args:
        key (types.PRNGKey): A random number generator.
        rays (struct.Rays): Batched rays of shape (...,).
        num_samples (int): Number of samples to be generated.
        near (float): Near plane.
        far (float): Far plane.
        use_randomized (bool): Add noise only if use_randomized is True.
        use_linear_disparity (bool): Use linear disparity.

    Returns:
        struct.Samples: Samples of shape (..., num_samples).
    """
    batch_shape = rays.origins.shape[:-1]

    if rays.near is not None:
        near = rays.near
    if rays.far is not None:
        far = rays.far

    t_vals = jnp.linspace(0.0, 1.0, num_samples)
    if not use_linear_disparity:
        tvals = near * (1.0 - t_vals) + far * t_vals
    else:
        tvals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    if use_randomized:
        tmids = 0.5 * (tvals[..., 1:] + tvals[..., :-1])
        upper = jnp.concatenate([tmids, tvals[..., -1:]], -1)
        lower = jnp.concatenate([tvals[..., :1], tmids], -1)
        trands = random.uniform(key, batch_shape + (num_samples,))
        tvals = lower + (upper - lower) * trands
    else:
        # Broadcast tvals to make the returned shape consistent.
        tvals = jnp.broadcast_to(tvals, batch_shape + (num_samples,))

    tvals = tvals[..., None]
    points = rays.origins[..., None, :] + tvals * rays.directions[..., None, :]
    return struct.Samples(
        xs=points,
        directions=rays.directions[..., None, :].repeat(num_samples, axis=-2),
        metadata=jax.tree_map(
            lambda x: x[..., None, :].repeat(num_samples, axis=-2),
            rays.metadata,
        )
        if rays.metadata is not None
        else None,
        tvals=tvals,
    )


def ipdf(
    key: types.PRNGKey,
    bins: jnp.ndarray,
    weights: jnp.ndarray,
    rays: struct.Rays,
    samples: struct.Samples,
    num_samples: int,
    use_randomized: bool,
) -> struct.Samples:
    """Hierarchical sampling.

    Args:
        key (types.PRNGKey): A random number generator.
        bins (jnp.ndarray): Bins of shape (..., n_bins + 1).
        weights (jnp.ndarray): Weights of shape (..., n_bins).
        rays (struct.rays): Rays of shape (...,).
        samples (struct.Samples): Samples of shape (..., num_samples_).
        num_samples (int): Number of samples to be generated.
        use_randomized (bool): Use stratified sampling.

    Returns:
        struct.Samples: Samples of shape (..., num_samples + num_samples_).
    """
    tvals = _piecewise_constant_pdf(
        key, bins, weights, num_samples, use_randomized
    )
    # FIXME: Get rid of this. Current repro requires this.
    # Also consider the old samples.
    tvals = jnp.sort(
        jnp.concatenate([tvals[..., None], samples.tvals], axis=-2), axis=-2
    )

    points = rays.origins[..., None, :] + tvals * rays.directions[..., None, :]
    return struct.Samples(
        xs=points,
        directions=rays.directions[..., None, :].repeat(
            tvals.shape[-2], axis=-2
        ),
        metadata=jax.tree_map(
            lambda x: x[..., None, :].repeat(tvals.shape[-2], axis=-2),
            rays.metadata,
        )
        if rays.metadata is not None
        else None,
        tvals=tvals,
    )
