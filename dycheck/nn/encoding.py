#!/usr/bin/env python3
#
# File   : embedding.py
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

import flax.linen as nn
import gin
import jax.numpy as jnp

from dycheck.utils import types

from .functional.posenc import posenc


@gin.configurable(denylist=["name"])
class Embed(nn.Module):
    """A shape-tolerant embedding layer.

    Attributes:
        num_embeddings (int): The number of embeddings.
        features: The dimensions of each embedding.
        embedding_init: The initializer to use for each.
    """

    num_embeddings: int = gin.REQUIRED
    features: int = gin.REQUIRED
    embedding_init: types.Activation = nn.initializers.uniform(scale=0.05)

    def setup(self):
        self.embed = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            embedding_init=self.embedding_init,
            name="embed",
        )

    @nn.compact
    def __call__(self, metadata: jnp.ndarray, **_) -> jnp.ndarray:
        """Method to get embeddings for specified indices.

        Args:
            metadata (jnp.ndarray): a (...,) or (..., 1) int array for
                embedding indices.

        Return:
            jnp.ndarray: a (..., D) array for queried embedding.
        """

        if metadata.shape[-1] == 1:
            metadata = jnp.squeeze(metadata, axis=-1)

        return self.embed(metadata)


@gin.configurable(denylist=["name"])
class PosEnc(nn.Module):
    """A positional encoding layer.

    Allow updating alpha during training.

    Example:
    .. code-block:: python
        pe = PosEnc(num_frames=8)

        # During training.
        ys, mutables = pe.apply(variables, xs, alpha=1.0, mutable=['alpha'])

        # During testing (use latest alpha from training).
        ys = pe.apply(variables, xs)
    """

    num_freqs: int = gin.REQUIRED
    use_identity: bool = False

    @nn.compact
    def __call__(
        self,
        xs: jnp.ndarray,
        alpha: Optional[float] = None,
        **_,
    ) -> jnp.ndarray:
        initializing = self.is_mutable_collection("params")
        alpha_var = self.variable(
            "alpha",
            "alpha",
            lambda shape: jnp.full(shape, self.num_freqs, jnp.float32),
            (1,),
        )

        if alpha is not None and not initializing:
            alpha_var.value = jnp.full((1,), alpha, jnp.float32)

        return posenc(
            xs, self.num_freqs, self.use_identity, alpha=alpha_var.value
        )


@gin.configurable(denylist=["name"])
class EmbedPosEnc(nn.Module):
    """A positional encoding layer that also embeds the input."""

    num_embeddings: int = gin.REQUIRED
    features: int = gin.REQUIRED
    embedding_init: types.Activation = nn.initializers.uniform(scale=0.05)

    num_freqs: int = gin.REQUIRED
    use_identity: bool = False

    def setup(self):
        self.embed = Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            embedding_init=self.embedding_init,
            name="embed",
        )
        self.posenc = PosEnc(
            num_freqs=self.num_freqs,
            use_identity=self.use_identity,
            name="posenc",
        )

    @nn.compact
    def __call__(
        self,
        xs: jnp.ndarray,
        metadata: jnp.ndarray,
        alpha: Optional[float] = None,
        **_,
    ) -> jnp.ndarray:
        return jnp.concatenate(
            [self.posenc(xs, alpha), self.embed(metadata)], axis=-1
        )
