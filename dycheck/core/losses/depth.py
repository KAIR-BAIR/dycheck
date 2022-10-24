#!/usr/bin/env python3
#
# File   : depth.py
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

from dycheck.nn import functional as F


@jax.jit
def compute_depth_loss(
    depth: jnp.ndarray, pred_depth: jnp.ndarray
) -> jnp.ndarray:
    loss = (pred_depth - depth) ** 2
    mask = (depth != 0).astype(jnp.float32)
    loss = F.common.masked_mean(loss, mask)
    loss = loss.mean()
    return loss
