#!/usr/bin/env python3
#
# File   : training.py
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
from collections import OrderedDict
from typing import Any, Dict, Tuple

import flax.linen as nn
import gin
import jax
import jax.numpy as jnp
from flax import core
from jax import random

import dycheck.nn.functional as F
from dycheck.core import metrics
from dycheck.utils import struct, types

from . import losses as losses_impl


@gin.configurable()
def train_step(
    model: nn.Module,
    key: types.PRNGKey,
    state: struct.TrainState,
    batch: Dict[str, Any],
    extra_params: struct.ExtraParams,
    train_scalars: struct.TrainScalars,
    *,
    use_random_bkgd: bool = False,
    use_bkgd_loss: bool = False,
    use_depth_loss: bool = False,
    use_dist_loss: bool = False,
    **_,
) -> Tuple[types.PRNGKey, struct.TrainState, Dict, Dict, Dict]:
    key, coarse_key, fine_key, reg_key = random.split(key, 4)

    def _level_fn(batch: Dict[str, Any], out: Dict[str, jnp.ndarray]):
        loss, stats = 0, OrderedDict()

        rgb_loss = F.common.masked_mean(
            (out["rgb"] - batch["rgb"]) ** 2, batch["mask"]
        )
        stats["loss/rgb"] = rgb_loss
        stats["metric/psnr"] = metrics.compute_psnr(
            out["rgb"], batch["rgb"], batch["mask"]
        )
        loss += rgb_loss

        if use_depth_loss:
            assert not model.use_sample_at_infinity, (
                "The original sampling at infinity trick will mess up with "
                "the depth optimization. Please disable it when applying "
                "depth loss."
            )
            depth = batch["depth"]
            pred_depth = out["depth"]
            # RGB mask is already merged into depth mask when preprocessing.
            depth_loss = losses_impl.compute_depth_loss(depth, pred_depth)
            stats["loss/depth"] = depth_loss
            loss += depth_loss * train_scalars.depth

        if use_dist_loss:
            assert not model.use_sample_at_infinity, (
                "The original sampling at infinity trick will mess up with "
                "the dist optimization. Please disable it when applying "
                "dist loss."
            )
            pred_weights = out["weights"]
            tvals = out["tvals"]
            pred_weights = pred_weights[..., :-1, :]
            svals = (tvals - model.near) / (model.far - model.near)
            dist_loss = losses_impl.compute_dist_loss(pred_weights, svals)
            stats["loss/dist"] = dist_loss
            loss += dist_loss * train_scalars.dist

        return loss, stats

    def _loss_fn(variables: core.FrozenDict):
        return_fields = ("rgb",)
        if use_depth_loss:
            return_fields += ("depth",)
        if use_dist_loss:
            return_fields += ("weights", "tvals")
        return_fields = tuple(set(return_fields))
        out, mutable = model.apply(
            variables,
            rays=batch["rays"],
            bkgd_rgb=None
            if not use_random_bkgd
            else random.uniform(reg_key, (3,), jnp.float32),
            extra_params=extra_params,
            return_fields=return_fields,
            mutable=["alpha"],
            rngs={"coarse": coarse_key, "fine": fine_key},
        )
        # Update the mutable alpha.
        variables = core.FrozenDict(
            {**mutable, **{k: v for k, v in variables.items() if k != "alpha"}}
        )

        losses, stats = {}, OrderedDict()

        if use_bkgd_loss:
            bkgd_points = batch["bkgd_points"]
            bkgd_loss = jax.jit(
                functools.partial(losses_impl.compute_bkgd_loss, model)
            )(key=reg_key, variables=variables, bkgd_points=bkgd_points)
            losses["bkgd"] = bkgd_loss
            stats["loss/bkgd"] = bkgd_loss * train_scalars.bkgd

        levels = ["coarse"] + ["fine"] if model.use_fine else []
        # Squash level stats into the main stats.
        for level in levels:
            losses[level], level_stats = _level_fn(batch, out[level])
            for k, v in level_stats.items():
                if k in stats:
                    stats[k] += v
                elif not k.startswith("metric/"):
                    stats[k] = v
                else:
                    stats[k + f"_{level}"] = v

        stats["loss/total"] = sum(losses.values())

        return stats["loss/total"], (variables, stats, out)

    optimizer = state.optimizer
    (_, (target, stats, out)), grad = jax.value_and_grad(
        _loss_fn, has_aux=True
    )(optimizer.target)
    # Replace the model variable in optimizer with the newly mutated ones. No
    # need for all-reduce bc mutable alphas are the same for all devices.
    optimizer = optimizer.replace(target=target)
    # All-reduce mean for updates over devices.
    stats, out, grad = jax.tree_map(
        functools.partial(jax.lax.pmean, axis_name="batch"),
        [stats, out, grad],
    )
    lr = train_scalars.lr
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
    state = state.replace(optimizer=optimizer)
    return key, state, stats, out, grad
