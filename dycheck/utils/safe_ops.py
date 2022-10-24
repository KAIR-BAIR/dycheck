#!/usr/bin/env python3
#
# File   : safe_ops.py
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

import functools
from typing import Tuple

import jax
import jax.numpy as jnp


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def safe_norm(
    x: jnp.ndarray,
    axis: int = -1,
    keepdims: bool = False,
    _: float = 1e-9,
) -> jnp.ndarray:
    """Calculates a np.linalg.norm(d) that's safe for gradients at d=0.

    These gymnastics are to avoid a poorly defined gradient for
    np.linal.norm(0). see https://github.com/google/jax/issues/3058 for details

    Args:
        x (jnp.ndarray): A jnp.array.
        axis (int): The axis along which to compute the norm.
        keepdims (bool): if True don't squeeze the axis.
        tol (float): the absolute threshold within which to zero out the
            gradient.

    Returns:
        Equivalent to np.linalg.norm(d)
    """
    return jnp.linalg.norm(x, axis=axis, keepdims=keepdims)


@safe_norm.defjvp
def _safe_norm_jvp(
    axis: int, keepdims: bool, tol: float, primals: Tuple, tangents: Tuple
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    (x,) = primals
    (x_dot,) = tangents
    safe_tol = max(tol, 1e-30)
    y = safe_norm(x, tol=safe_tol, axis=axis, keepdims=True)
    y_safe = jnp.maximum(y, tol)  # Prevent divide by zero.
    y_dot = jnp.where(y > safe_tol, x_dot * x / y_safe, jnp.zeros_like(x))
    y_dot = jnp.sum(y_dot, axis=axis, keepdims=True)
    # Squeeze the axis if `keepdims` is True.
    if not keepdims:
        y = jnp.squeeze(y, axis=axis)
        y_dot = jnp.squeeze(y_dot, axis=axis)
    return y, y_dot


def log1p_safe(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.minimum(x, 3e37))


def exp_safe(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(jnp.minimum(x, 87.5))


def expm1_safe(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.expm1(jnp.minimum(x, 87.5))


def safe_sqrt(x: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    safe_x = jnp.where(x == 0, jnp.ones_like(x) * eps, x)
    return jnp.sqrt(safe_x)
