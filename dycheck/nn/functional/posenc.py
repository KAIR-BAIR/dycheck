#!/usr/bin/env python3
#
# File   : posenc.py
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


def _posenc_window(num_freqs: int, alpha: jnp.ndarray) -> jnp.ndarray:
    """Windows a posenc using a cosiney window.

    This is equivalent to taking a truncated Hann window and sliding it to the
    right along the frequency spectrum.

    Args:
        num_freqs (int): The number of frequencies in the posenc.
        alpha (jnp.ndarray): The maximal frequency that allows by the window.

    Returns:
        jnp.ndarray: A (..., num_freqs) array of window values.
    """
    freqs = jnp.arange(num_freqs, dtype=jnp.float32)
    xs = jnp.clip(alpha - freqs, 0, 1)
    return 0.5 * (1 + jnp.cos(jnp.pi * xs + jnp.pi))


def posenc(
    xs: jnp.ndarray,
    num_freqs: int,
    use_identity: bool = False,
    alpha: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Positional encoding using sinusoidal bases.

    Args:
        xs (jnp.ndarray): A (..., C) array of positions.
        num_freqs (int): The number of sinusoidal frequencies to use.
        use_identity (bool): If True, prepends the identity to the encoding.
        alpha (jnp.ndarray): If not None, will use a cosine window to anneal the
            encoding.

    Returns:
        four_feats (jnp.ndarray): A (..., 2F * C) array of sinusoidal encodings
            if use_identity is False, otherwise a (..., 2F * C + 1) array.
    """
    batch_shape = xs.shape[:-1]

    scales = 2.0 ** jnp.arange(num_freqs)
    # (..., F, C).
    xb = xs[..., None, :] * scales[:, None]
    # (..., F, 2, C).
    four_feats = jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], axis=-2))

    if alpha is not None:
        window = _posenc_window(num_freqs, alpha)
        four_feats = window[..., None, None] * four_feats

    # (*, 2F * C).
    four_feats = four_feats.reshape((*batch_shape, -1))

    if use_identity:
        return jnp.concatenate([xs, four_feats], axis=-1)
    else:
        return four_feats
