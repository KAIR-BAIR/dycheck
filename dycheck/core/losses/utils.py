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

import jax
import jax.numpy as jnp

from dycheck.utils import safe_ops


@jax.jit
def general_loss_with_squared_residual(
    squared_x: jnp.ndarray, alpha: float, scale: float
) -> jnp.ndarray:
    """The general loss that takes a squared residual.

    This fuses the sqrt operation done to compute many residuals while
    preserving the square in the loss formulation.

    This implements the rho(x, alpha, c) function described in "A General and
    Adaptive Robust Loss Function", Jonathan T. Barron,
    https://arxiv.org/abs/1701.03077.

    Args:
        squared_x (jnp.ndarray): The residual for which the loss is being
            computed. x can have any shape, and alpha and scale will be
            broadcasted to match x's shape if necessary.
        alpha (float): The shape parameter of the loss (alpha in the paper),
            where more negative values produce a loss with more robust behavior
            (outliers "cost" less), and more positive values produce a loss
            with less robust behavior (outliers are penalized more heavily).
            Alpha can be any value in [-infinity, infinity], but the gradient
            of the loss with respect to alpha is 0 at -infinity, infinity, 0,
            and 2. Varying alpha allows for smooth interpolation between
            several discrete robust losses:
                alpha=-Infinity: Welsch/Leclerc Loss.
                alpha=-2: Geman-McClure loss.
                alpha=0: Cauchy/Lortentzian loss.
                alpha=1: Charbonnier/pseudo-Huber loss.
                alpha=2: L2 loss.
        scale (float): The scale parameter of the loss. When |x| < scale, the
            loss is an L2-like quadratic bowl, and when |x| > scale the loss
            function takes on a different shape according to alpha.

    Returns:
        jnp.ndarray: The losses for each element of x, in the same shape as x.
    """
    eps = jnp.finfo(jnp.float32).eps

    # This will be used repeatedly.
    squared_scaled_x = squared_x / (scale**2)

    # The loss when alpha == 2.
    loss_two = 0.5 * squared_scaled_x
    # The loss when alpha == 0.
    loss_zero = safe_ops.log1p_safe(0.5 * squared_scaled_x)
    # The loss when alpha == -infinity.
    loss_neginf = -jnp.expm1(-0.5 * squared_scaled_x)
    # The loss when alpha == +infinity.
    loss_posinf = safe_ops.expm1_safe(0.5 * squared_scaled_x)

    # The loss when not in one of the above special cases.
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = jnp.maximum(eps, jnp.abs(alpha - 2.0))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = jnp.where(
        jnp.greater_equal(alpha, 0.0),
        jnp.ones_like(alpha),
        -jnp.ones_like(alpha),
    ) * jnp.maximum(eps, jnp.abs(alpha))
    loss_otherwise = (beta_safe / alpha_safe) * (
        jnp.power(squared_scaled_x / beta_safe + 1.0, 0.5 * alpha) - 1.0
    )

    # Select which of the cases of the loss to return.
    loss = scale * jnp.where(
        alpha == -jnp.inf,
        loss_neginf,
        jnp.where(
            alpha == 0,
            loss_zero,
            jnp.where(
                alpha == 2,
                loss_two,
                jnp.where(alpha == jnp.inf, loss_posinf, loss_otherwise),
            ),
        ),
    )

    return loss
