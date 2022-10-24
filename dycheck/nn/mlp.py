#!/usr/bin/env python3
#
# File   : mlp.py
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

from typing import Literal, Optional, Tuple

import gin
import jax
import jax.numpy as jnp
from flax import linen as nn

from dycheck.nn import functional as F
from dycheck.utils import common, types


@gin.configurable(denylist=["name"])
class MLP(nn.Module):
    # FIXME: This MLP has different skip scheme compared to all the other
    # NeRFs. Repro however requires it.

    depth: int = gin.REQUIRED
    width: int = gin.REQUIRED

    hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
    hidden_activation: types.Activation = nn.relu

    output_init: types.Initializer = jax.nn.initializers.glorot_uniform()
    output_channels: int = 0
    output_activation: types.Activation = F.activations.identity

    use_bias: bool = True
    skips: Tuple[int] = tuple()

    def setup(self):
        # NOTE(Hang Gao @ 03/06): Do static setup rather than compact because
        # we might use this function for root-finding, which requires this part
        # to be pre-declared.
        layers = [
            nn.Dense(
                self.width,
                use_bias=self.use_bias,
                kernel_init=self.hidden_init,
                name=f"hidden_{i}",
            )
            for i in range(self.depth)
        ]
        self.layers = layers

        if self.output_channels > 0:
            self.logit_layer = nn.Dense(
                self.output_channels,
                use_bias=self.use_bias,
                kernel_init=self.output_init,
                name="logit",
            )

    def __call__(self, xs: jnp.ndarray) -> jnp.ndarray:
        inputs = xs
        for i in range(self.depth):
            layer = self.layers[i]
            if i in self.skips:
                xs = jnp.concatenate([xs, inputs], axis=-1)
            xs = layer(xs)
            xs = self.hidden_activation(xs)

        if self.output_channels > 0:
            xs = self.logit_layer(xs)
            xs = self.output_activation(xs)

        return xs


@gin.configurable(denylist=["name"])
class NeRFMLP(nn.Module):
    trunk_depth: int = 8
    trunk_width: int = 256

    sigma_depth: int = 0
    sigma_width: int = 128
    sigma_channels: int = 1

    rgb_depth: int = 1
    rgb_width: int = 128
    rgb_channels: int = 3

    hidden_activation: types.Activation = nn.relu
    skips: Tuple[int] = (4,)

    @nn.compact
    def __call__(
        self,
        xs: jnp.ndarray,
        trunk_conditions: Optional[jnp.ndarray] = None,
        rgb_conditions: Optional[jnp.ndarray] = None,
        return_fields: Tuple[Literal["point_sigma", "point_rgb"]] = (
            "point_sigma",
            "point_rgb",
        ),
    ) -> jnp.ndarray:
        trunk_mlp = MLP(
            depth=self.trunk_depth,
            width=self.trunk_width,
            hidden_activation=self.hidden_activation,
            skips=self.skips,
            name="trunk",
        )
        sigma_mlp = MLP(
            depth=self.sigma_depth,
            width=self.sigma_width,
            hidden_activation=self.hidden_activation,
            output_channels=self.sigma_channels,
            name="sigma",
        )
        rgb_mlp = MLP(
            depth=self.rgb_depth,
            width=self.rgb_width,
            hidden_activation=self.hidden_activation,
            output_channels=self.rgb_channels,
            name="rgb",
        )

        trunk = xs
        if trunk_conditions is not None:
            trunk = jnp.concatenate([trunk, trunk_conditions], axis=-1)
        trunk = trunk_mlp(trunk)

        sigma = sigma_mlp(trunk)
        #  print(sigma.mean(), trunk.mean(), xs.mean())

        if rgb_conditions is not None:
            # Use one extra layer to align with original NeRF model.
            trunk = nn.Dense(
                trunk_mlp.width,
                kernel_init=trunk_mlp.hidden_init,
                name=f"bottleneck",
            )(trunk)
            trunk = jnp.concatenate([trunk, rgb_conditions], axis=-1)
        rgb = rgb_mlp(trunk)

        out = {"point_sigma": sigma, "point_rgb": rgb}

        out = common.traverse_filter(
            out, return_fields=return_fields, inplace=True
        )
        return out
