#!/usr/bin/env python3
#
# File   : ambient.py
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
from typing import Callable, Dict, Optional

import gin
import jax
import jax.numpy as jnp
from flax import linen as nn

from dycheck.geometry import matv, se3
from dycheck.nn import MLP, PosEnc
from dycheck.utils import struct, types

from .dense import TranslDensePosEnc


@gin.configurable(denylist=["name"])
class TranslAmbientPosEnc(TranslDensePosEnc):
    ambient_cls: Callable[..., nn.Module] = functools.partial(
        MLP,
        depth=6,
        width=64,
        output_init=jax.nn.initializers.normal(1e-5),
        output_channels=2,
        skips=(4,),
    )
    ambient_embed_cls: Callable[..., nn.Module] = functools.partial(
        PosEnc,
        num_freqs=6,
        use_identity=False,
    )
    ambient_points_embed_cls: Callable[..., nn.Module] = functools.partial(
        PosEnc,
        num_freqs=1,
        use_identity=False,
    )

    def setup(self):
        super().setup()

        self.ambient_embed = self.ambient_embed_cls()
        self.ambient_points_embed = self.ambient_points_embed_cls()
        self.ambient = self.ambient_cls(hidden_init=self.hidden_init)

    def _eval(
        self,
        xs: jnp.ndarray,
        metadata: struct.Metadata,
        extra_params: Optional[struct.ExtraParams],
        use_warped_points_embed: bool = True,
        ambient_points: Optional[jnp.ndarray] = None,
        **_,
    ) -> Dict[str, jnp.ndarray]:
        assert self.points_embed_key in metadata._fields
        metadata = getattr(metadata, self.points_embed_key)
        points_embed = self.points_embed(
            xs=xs,
            metadata=metadata,
            alpha=getattr(extra_params, "warp_alpha")
            if extra_params
            else None,
        )
        warped_points = self.trunk(points_embed) + xs

        if ambient_points is None:
            metadata_embed = points_embed[..., -self.points_embed.features :]
            ambient_embed = self.ambient_embed(xs)
            ambient_points = self.ambient(
                jnp.concatenate([ambient_embed, metadata_embed], axis=-1)
            )

        out = {
            "warped_points": warped_points,
            "ambient_points": ambient_points,
        }
        if use_warped_points_embed:
            out["warped_points_embed"] = jnp.concatenate(
                [
                    self.warped_points_embed(warped_points),
                    self.ambient_points_embed(
                        ambient_points,
                        alpha=getattr(extra_params, "ambient_alpha")
                        if extra_params
                        else None,
                    ),
                ],
                axis=-1,
            )
        return out


@gin.configurable(denylist=["name"])
class SE3AmbientPosEnc(TranslAmbientPosEnc):
    trunk_cls: Callable[..., nn.Module] = functools.partial(
        MLP,
        depth=6,
        width=128,
        skips=(4,),
    )

    rotation_depth: int = 0
    rotation_width: int = 128
    rotation_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)

    transl_depth: int = 0
    transl_width: int = 128
    transl_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)

    hidden_init: types.Initializer = jax.nn.initializers.xavier_uniform()

    def setup(self):
        super().setup()

        self.branches = {
            "rotation": MLP(
                depth=self.rotation_depth,
                width=self.rotation_width,
                hidden_init=self.hidden_init,
                output_init=self.rotation_init,
                output_channels=3,
            ),
            "transl": MLP(
                depth=self.transl_depth,
                width=self.transl_width,
                hidden_init=self.hidden_init,
                output_init=self.transl_init,
                output_channels=3,
            ),
        }

    def _eval(
        self,
        xs: jnp.ndarray,
        metadata: struct.Metadata,
        extra_params: Optional[struct.ExtraParams],
        use_warped_points_embed: bool = True,
        ambient_points: Optional[jnp.ndarray] = None,
        **_,
    ) -> Dict[str, jnp.ndarray]:
        assert self.points_embed_key in metadata._fields
        metadata = getattr(metadata, self.points_embed_key)
        points_embed = self.points_embed(
            xs=xs,
            metadata=metadata,
            alpha=getattr(extra_params, "warp_alpha")
            if extra_params
            else None,
        )
        trunk = self.trunk(points_embed)

        rotation = self.branches["rotation"](trunk)
        transl = self.branches["transl"](trunk)
        theta = jnp.linalg.norm(rotation, axis=-1)
        rotation = rotation / theta[..., None]
        transl = transl / theta[..., None]
        screw_axis = jnp.concatenate([rotation, transl], axis=-1)
        transform = se3.exp_se3(screw_axis, theta)

        warped_points = se3.from_homogenous(
            matv(transform, se3.to_homogenous(xs))
        )

        if ambient_points is None:
            metadata_embed = points_embed[..., -self.points_embed.features :]
            ambient_embed = self.ambient_embed(xs)
            ambient_points = self.ambient(
                jnp.concatenate([ambient_embed, metadata_embed], axis=-1)
            )

        out = {
            "warped_points": warped_points,
            "ambient_points": ambient_points,
        }
        if use_warped_points_embed:
            out["warped_points_embed"] = jnp.concatenate(
                [
                    self.warped_points_embed(warped_points),
                    self.ambient_points_embed(
                        ambient_points,
                        alpha=getattr(extra_params, "ambient_alpha")
                        if extra_params
                        else None,
                    ),
                ],
                axis=-1,
            )
        return out
