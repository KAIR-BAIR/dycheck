#!/usr/bin/env python3
#
# File   : warping.py
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

import jax
import jax.numpy as jnp
import numpy as np
from flax import core
from flax import linen as nn
from jax import random

from dycheck import geometry
from dycheck.nn import functional as F
from dycheck.utils import common, struct, types


def get_pwarp_points(
    model: nn.Module,
    return_fields: Sequence[str] = (
        "warped_points",
        "diffs",
        "converged",
    ),
    **kwargs,
) -> Callable:
    """Get the pmap'd warping function for point transform by passing NeRF
    model and its arguments.

    Args:
        model (nn.Module): The base NeRF model.
        return_fields (Sequence[str]): The fields to return.

    Returns:
        Callable: The pmap'd warping function.
    """

    @F.unshard_pmap_wrapper(
        in_axes=(0, 0),
        donate_argnums=(3,),
        axis_name="batch",
    )
    def _model_fn(
        variables: core.FrozenDict, samples: struct.Samples
    ) -> Dict[str, jnp.ndarray]:
        out = model.apply(
            variables,
            samples,
            extra_params=None,
            return_fields=return_fields,
            method=lambda m, *args, **kwargs: m.points_embed(*args, **kwargs),
            **kwargs,
        )
        return out

    return functools.partial(_warp_points, _model_fn)


def _warp_points(
    _model_fn: Callable[
        [core.FrozenDict, struct.Samples],
        Dict[str, jnp.ndarray],
    ],
    variables: core.FrozenDict,
    samples: struct.Samples,
    *,
    chunk: int = 8192,
    show_pbar: bool = True,
    desc: str = "* Warping points",
    pbar_kwargs: Dict[str, Any] = {},
    **_,
) -> Dict[str, jnp.ndarray]:
    """Warp all the points from one frame to the another frame to form an image
    through NeRF model.

    Note that this functions warps points in 3D.

    Assume one process/host only. Can be made more generic but it is not used
    now.

    Args:
        _model_fn (...): The model function. Would be taken care of internally
            and user should never specify it.
        variables (core.FrozenDict): The model parameters. Assume it is already
            replicated.
        samples (struct.Samples): The samples to warp of shape (...).
        chunk (int): The chunk size for rendering.
        show_pbar (bool): Whether to show the progress bar.

    Returns:
        Dict[str, jnp.ndarray]: The warped points of the original shape (...).
    """
    assert jax.process_count() == 1, "Only one process/host is supported."
    num_devices = jax.local_device_count()

    batch_shape = samples.xs.shape[:-1]
    num_samples = np.prod(batch_shape)

    samples = jax.tree_map(lambda x: x.reshape((num_samples, -1)), samples)

    results = []
    for i in (common.tqdm if show_pbar else lambda x, **_: x)(
        range(0, num_samples, chunk), desc=desc, **pbar_kwargs
    ):
        chunk_slice_fn = lambda x: x[i : i + chunk]
        chunk_samples = jax.tree_map(chunk_slice_fn, samples)
        num_chunk_samples = chunk_samples.xs.shape[0]
        remainder = num_chunk_samples % num_devices
        if remainder != 0:
            padding = num_devices - remainder
            chunk_pad_fn = lambda x: jnp.pad(
                x, ((0, padding), (0, 0)), mode="edge"
            )
            chunk_samples = jax.tree_map(chunk_pad_fn, chunk_samples)
        else:
            padding = 0
        out = _model_fn(variables, F.shard(chunk_samples))
        out = jax.tree_map(lambda x: x[: x.shape[0] - padding], out)
        results.append(out)

    results = jax.tree_multimap(
        lambda *x: jnp.concatenate(x, axis=0), *results
    )
    results = jax.tree_map(
        lambda x: x.reshape(batch_shape + x.shape[1:]), results
    )

    return results


def get_pwarp_pixels(
    model: nn.Module,
    use_randomized: bool = False,
    return_fields: Sequence[str] = (
        "warped_pixels",
        "diffs",
        "converged",
    ),
    **kwargs,
) -> Callable:
    """Get the pmap'd warping function for pixel transform by passing NeRF
    model and its arguments.

    Args:
        model (nn.Module): The base NeRF model.
        return_fields (Sequence[str]): The fields to return.

    Returns:
        Callable: The pmap'd warping function.
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
        assert rays.metadata is not None
        metadata = rays.metadata
        rays = rays._replace(metadata=metadata._replace(time_to=None))

        # Get the points and their weights from NeRF.
        rendered_out = model.apply(
            variables,
            rays,
            extra_params=None,
            use_randomized=use_randomized,
            return_fields=return_fields + ("weights", "points"),
            #  return_fields=return_fields
            #  + ("weights", "points", "warp_out/cano_points"),
            **kwargs,
            rngs=rngs,
        )
        rendered_out = (
            rendered_out["fine"]
            if rendered_out["fine"]
            else rendered_out["coarse"]
        )

        # Warp-integrate; project is left to the outer loop function since
        # camera obj does not play nicely with pmap right now.
        samples = struct.Samples(
            xs=rendered_out["points"],
            directions=None,
            metadata=jax.tree_map(
                lambda x: x[..., None, :].repeat(
                    rendered_out["points"].shape[-2], axis=-2
                ),
                metadata,
            ),
        )
        warped_out = model.apply(
            variables,
            samples,
            extra_params=None,
            return_fields=return_fields + ("warped_points",),
            method=lambda m, *args, **kwargs: m.points_embed(*args, **kwargs),
            **kwargs,
        )

        assert (
            "weights" not in return_fields
        ), "Returning weights is currently not supported."
        out = common.traverse_filter(
            {**rendered_out, **warped_out},
            return_fields=return_fields + ("warped_points",),
        )
        out = jax.tree_map(
            lambda x: (x * rendered_out["weights"]).sum(axis=-2), out
        )
        #  out = {**rendered_out, **warped_out, **{"samples": samples}}
        return out

    return functools.partial(_warp_pixels, _model_fn)


def _warp_pixels(
    _model_fn: Callable[
        [core.FrozenDict, struct.Samples],
        Dict[str, jnp.ndarray],
    ],
    variables: core.FrozenDict,
    pixels: jnp.ndarray,
    metadata: struct.Metadata,
    camera: geometry.Camera,
    camera_to: geometry.Camera,
    *,
    key: types.PRNGKey,
    chunk: int = 8192,
    show_pbar: bool = True,
    desc: str = "* Warping pixels",
    pbar_kwargs: Dict[str, Any] = {},
    **_,
) -> Dict[str, jnp.ndarray]:
    """Warp all the points from one frame to the another frame to form an image
    through NeRF model.

    Assume one process/host only. Can be made more generic but it is not used
    now.

    Args:
        _model_fn (...): The model function. Would be taken care of internally
            and user should never specify it.
        variables (core.FrozenDict): The model parameters. Assume it is already
            replicated.
        pixels (jnp.ndarray): The pixels to warp of shape (..., 2).
        metadata (struct.Metadata): The metadata of the pixels of shape (...,).
            Note that time and time_to must be specified.
        camera (geometry.Camera): The camera that the pixels belong to.
        camera_to (geometry.Camera): The camera that the pixels to warp to.
        chunk (int): The chunk size for rendering.
        show_pbar (bool): Whether to show the progress bar.

    Returns:
        Dict[str, jnp.ndarray]: The warped pixels of the original shape (...,
            2).
    """
    assert (
        metadata.time is not None and metadata.time_to is not None
    ), "Metadata has not specified time and time_to."

    assert jax.process_count() == 1, "Only one process/host is supported."
    num_devices = jax.local_device_count()

    rays = camera.pixels_to_rays(pixels)._replace(metadata=metadata)

    batch_shape = pixels.shape[:-1]
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
        out = jax.tree_map(lambda x: x[: x.shape[0] - padding], out)
        results.append(out)

    results = jax.tree_multimap(
        lambda *x: jnp.concatenate(x, axis=0), *results
    )
    results = jax.tree_map(
        lambda x: x.reshape(batch_shape + x.shape[1:]), results
    )
    results["warped_pixels"] = geometry.project(
        results.pop("warped_points"),
        jnp.array(camera_to.intrin, jnp.float32),
        jnp.array(camera_to.extrin, jnp.float32),
        jnp.array(camera_to.radial_distortion, jnp.float32),
        jnp.array(camera_to.tangential_distortion, jnp.float32),
    )

    return results
