#!/usr/bin/env python3
#
# File   : se3.py
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
from jax import numpy as jnp

from . import utils


@jax.jit
def skew(w: jnp.ndarray) -> jnp.ndarray:
    """Build a skew matrix ("cross product matrix") for vector w.
    Modern Robotics Eqn 3.30.

    Args:
        w: (..., 3,) A 3-vector

    Returns:
        W: (..., 3, 3) A skew matrix such that W @ v == w x v
    """
    zeros = jnp.zeros_like(w[..., 0])
    return jnp.stack(
        [
            jnp.stack([zeros, -w[..., 2], w[..., 1]], axis=-1),
            jnp.stack([w[..., 2], zeros, -w[..., 0]], axis=-1),
            jnp.stack([-w[..., 1], w[..., 0], zeros], axis=-1),
        ],
        axis=-2,
    )


def rt_to_se3(R: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Rotation and translation to homogeneous transform.

    Args:
        R: (..., 3, 3) An orthonormal rotation matrix.
        t: (..., 3,) A 3-vector representing an offset.

    Returns:
        X: (..., 4, 4) The homogeneous transformation matrix described by
            rotating by R and translating by t.
    """
    batch_shape = R.shape[:-2]
    return jnp.concatenate(
        [
            jnp.concatenate([R, t[..., None]], axis=-1),
            jnp.broadcast_to(
                jnp.array([[0, 0, 0, 1]], jnp.float32), batch_shape + (1, 4)
            ),
        ],
        axis=-2,
    )


def exp_so3(w: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

    Args:
        w: (..., 3,) An axis of rotation. This is assumed to be a unit-vector.
        theta (...,): An angle of rotation.

    Returns:
        R: (..., 3, 3) An orthonormal rotation matrix representing a rotation
            of magnitude theta about axis w.
    """
    batch_shape = w.shape[:-1]
    W = skew(w)
    return (
        jnp.broadcast_to(jnp.eye(3), batch_shape + (3, 3))
        + jnp.sin(theta)[..., None, None] * W
        + (1 - jnp.cos(theta)[..., None, None]) * utils.matmul(W, W)
    )


def exp_se3(S: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.88.

    Args:
      S: (..., 6,) A screw axis of motion.
      theta (...,): Magnitude of motion.

    Returns:
      a_X_b: (..., 4, 4) The homogeneous transformation matrix attained by
          integrating motion of magnitude theta about S for one second.
    """
    batch_shape = S.shape[:-1]
    w, v = jnp.split(S, 2, axis=-1)
    W = skew(w)
    R = exp_so3(w, theta)
    t = utils.matv(
        (
            theta[..., None, None]
            * jnp.broadcast_to(jnp.eye(3), batch_shape + (3, 3))
            + (1 - jnp.cos(theta)[..., None, None]) * W
            + (theta[..., None, None] - jnp.sin(theta)[..., None, None])
            * utils.matmul(W, W)
        ),
        v,
    )
    return rt_to_se3(R, t)


def to_homogenous(v):
    return jnp.concatenate([v, jnp.ones_like(v[..., :1])], axis=-1)


def from_homogenous(v):
    return v[..., :3] / v[..., -1:]
