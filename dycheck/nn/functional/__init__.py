#!/usr/bin/env python3
#
# File   : __init__.py
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

from typing import Callable, Dict

import jax
import jax.numpy as jnp

from dycheck.utils import types

from . import activations, broyden, common, posenc, rendering, sampling


def to_device(xs: Dict[str, types.Array]) -> Dict[str, jnp.ndarray]:
    return jax.tree_map(jnp.array, xs)


def shard(xs: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    device_count = jax.local_device_count()
    return jax.tree_map(
        lambda x: x.reshape((device_count, -1) + x.shape[1:]), xs
    )


def unshard(xs: Dict[str, jnp.ndarray], padding: int = 0) -> jnp.ndarray:
    if padding > 0:
        fn = lambda xs: xs.reshape(
            [xs.shape[0] * xs.shape[1]] + list(xs.shape[2:])
        )[:-padding]
    else:
        fn = lambda xs: xs.reshape(
            [xs.shape[0] * xs.shape[1]] + list(xs.shape[2:])
        )
    return jax.tree_map(fn, xs)


def unshard_pmap_wrapper(**kwargs) -> Callable:
    def wrapped(fn: Callable) -> Callable:
        fn = jax.pmap(fn, **kwargs)
        return lambda *wrapped_args, **wrapped_kwargs: unshard(
            fn(*wrapped_args, **wrapped_kwargs)
        )

    return wrapped
