#!/usr/bin/env python3
#
# File   : trainer.py
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
import time
from typing import Any, Callable, Dict, Tuple

import gin
import jax
import jax.numpy as jnp
from absl import logging
from flax import jax_utils
from jax import random

from dycheck.utils import struct, types

from .base import Engine


@gin.configurable()
class Trainer(Engine):
    training: bool = True

    def __init__(
        self,
        max_steps: int = gin.REQUIRED,
        train_step: Callable[
            [
                types.PRNGKey,
                struct.TrainState,
                Dict[str, Any],
                struct.ExtraParams,
                struct.TrainScalars,
            ],
            Tuple[
                types.PRNGKey,
                struct.TrainState,
                Dict[str, Any],
                Dict[str, jnp.ndarray],
            ],
        ] = gin.REQUIRED,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.train_step = functools.partial(train_step, self.model)

    def launch(self):
        self.tasks.start()
        self.schedules = struct.TrainSchedules()

        logging.info("* Starting training.")

        self.pdataset = jax_utils.prefetch_to_device(self.dataset, 3)
        self.pstate = jax_utils.replicate(self.state)
        self.pkeys = random.split(self.key, jax.local_device_count())
        self.ptrain_step = jax.pmap(
            self.train_step,
            in_axes=(0, 0, 0, 0, None),
            axis_name="batch",
            donate_argnums=(2,),
        )

        self.start_time = time.time()
        for step, pbatch in zip(
            range(self.init_step, self.max_steps + 1), self.pdataset
        ):
            self.step = step
            self.pbatch = pbatch

            # Training logic.
            self.scalars = self.schedules.eval_scalars(step)
            self.extra_params = self.schedules.eval_extra_params(step)
            self.pextra_params = jax.tree_map(
                lambda x: x[..., None],
                jax_utils.replicate(
                    self.extra_params,
                ),
            )
            (
                self.pkeys,
                self.pstate,
                self.pstats,
                self.pout,
                self.pgrad,
            ) = self.ptrain_step(
                self.pkeys,
                self.pstate,
                self.pbatch,
                self.pextra_params,
                self.scalars,
            )

            self.tasks.every_n_steps()

        self.tasks.finalize()
