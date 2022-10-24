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

import functools
from typing import Any, Callable, Dict, Literal, Sequence

import gin
import jax
import jax.numpy as jnp
import numpy as np
from flax import core
from flax import linen as nn
from jax import random

from dycheck.nn import functional as F
from dycheck.utils import common, struct, types


@gin.configurable()
def get_prender_image(
    model: nn.Module,
    use_randomized: bool = False,
    return_fields: Sequence[str] = (
        "rgb",
        "depth",
        "acc",
    ),
    chunk: int = 8192,
    **kwargs,
) -> Callable:
    """Get the pmap'd rendering function by passing NeRF model and its
    arguments.

    Args:
        model (nn.Module): The model to render.
        use_randomized (bool): Whether to use randomized parameters.
        return_fields (Sequence[str]): The fields to return.

    Returns:
        Callable: The pmap'd rendering function.
    """

    @F.unshard_pmap_wrapper(
        in_axes=(0, 0, 0),
        donate_argnums=(3,),
        axis_name="batch",
    )
    def _model_fn(
        variables: core.FrozenDict,
        rays: struct.Rays,
        rngs: Dict[Literal["coarse", "fine"], types.PRNGKey],
    ) -> Dict[str, Dict[str, jnp.ndarray]]:
        out = model.apply(
            variables,
            rays,
            extra_params=None,
            use_randomized=use_randomized,
            return_fields=return_fields,
            **kwargs,
            rngs=rngs,
        )
        return out

    return functools.partial(_render_image, _model_fn, chunk=chunk)


def _render_image(
    _model_fn: Callable[
        [
            core.FrozenDict,
            struct.Rays,
            Dict[Literal["coarse", "fine"], types.PRNGKey],
        ],
        Dict[str, Dict[str, jnp.ndarray]],
    ],
    variables: core.FrozenDict,
    rays: struct.Rays,
    *,
    key: types.PRNGKey,
    chunk: int = 8192,
    show_pbar: bool = True,
    desc: str = "* Rendering image",
    pbar_kwargs: Dict[str, Any] = {},
    **_,
) -> Dict[str, jnp.ndarray]:
    """Render all the rays to form an image through NeRF model.

    Assume one process/host only. Can be made more generic but it is not used
    now.

    Args:
        _model_fn (...): The model function. Would be taken care of internally
            and user should never specify it.
        variables (core.FrozenDict): The model parameters. Assume it is already
            replicated.
        rays (struct.Rays): The rays to render of shape (H, W).
        key (types.PRNGKey): The PRNG key.
        chunk (int): The chunk size for rendering.
        show_pbar (bool): Whether to show the progress bar.

    Returns:
        Dict[str, jnp.ndarray]: The rendered image.
    """
    assert jax.process_count() == 1, "Only one process/host is supported."
    num_devices = jax.local_device_count()

    batch_shape = rays.origins.shape[:-1]
    num_rays = np.prod(batch_shape)

    rays = jax.tree_map(lambda x: x.reshape((num_rays, -1)), rays)

    _, key0, key1 = random.split(key, 3)
    rngs = jax.tree_map(
        lambda x: random.split(x, num_devices), {"coarse": key0, "fine": key1}
    )

    results = []
    for i in (common.tqdm if show_pbar else lambda x, **_: x)(
        range(0, num_rays, chunk), desc=desc, **pbar_kwargs
    ):
        chunk_slice_fn = lambda x: x[i : i + chunk]
        chunk_rays = jax.tree_map(chunk_slice_fn, rays)
        num_chunk_rays = chunk_rays.origins.shape[0]
        remainder = num_chunk_rays % num_devices
        if remainder != 0:
            padding = num_devices - remainder
            chunk_pad_fn = lambda x: jnp.pad(
                x, ((0, padding), (0, 0)), mode="edge"
            )
            chunk_rays = jax.tree_map(chunk_pad_fn, chunk_rays)
        else:
            padding = 0
        out = _model_fn(variables, F.shard(chunk_rays), rngs)
        out = out["fine"] if out["fine"] else out["coarse"]
        out = jax.tree_map(lambda x: x[: x.shape[0] - padding], out)
        results.append(out)

    results = jax.tree_multimap(
        lambda *x: jnp.concatenate(x, axis=0), *results
    )
    results = jax.tree_map(
        lambda x: x.reshape(batch_shape + x.shape[1:]), results
    )

    return results
