#!/usr/bin/env python3
#
# File   : dump.py
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

import gin
import jax
from absl import logging
from flax.errors import InvalidCheckpointError
from flax.training import checkpoints

from dycheck.utils import path_ops

from . import base


@gin.configurable(denylist=["engine"])
class Checkpoint(base.Task):
    """Dump checkpoints periodically during training."""

    @property
    def eligible(self):
        return self.engine.training

    def start(self):
        pass

    def every_n_steps(self):
        self._dump_checkpoint()

    def finalize(self):
        try:
            self._dump_checkpoint()
        except InvalidCheckpointError:
            logging.info("* Checkpoint already exists. Skipping dump.")
            pass

    def _dump_checkpoint(self):
        engine = self.engine

        logging.info("* Dumping checkpoint.")
        state = jax.device_get(jax.tree_map(lambda x: x[0], engine.pstate))
        step = state.optimizer.state.step

        path_ops.mkdir(engine.checkpoint_dir)
        checkpoints.save_checkpoint(engine.checkpoint_dir, state, step, keep=5)

        engine.state = state
