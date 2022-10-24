#!/usr/bin/env python3
#
# File   : base.py
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

import os.path as osp
from typing import Callable, Optional, Tuple

import gin
import jax.numpy as jnp
from absl import logging
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from jax import random

from dycheck.datasets import Dataset
from dycheck.utils import common, struct, types

from ..tasks import Tasks


class Engine(object):
    def __init__(
        self,
        dataset_cls: Callable[..., Dataset] = gin.REQUIRED,
        model_cls: Callable[..., nn.Module] = gin.REQUIRED,
        *,
        name: str = "engine",
        work_root: Optional[types.PathType] = None,
        checkpoint_step: Optional[int] = None,
        checkpoint_path: Optional[types.PathType] = None,
        **_,
    ):
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls

        self.name = name
        self.work_root = work_root or osp.abspath(
            osp.join(osp.dirname(__file__), "..", "..", "..", "work_dirs")
        )
        self.checkpoint_step = checkpoint_step
        self.checkpoint_path = checkpoint_path

        self.dataset = self.build_dataset()

        self.work_dir = osp.join(
            self.work_root,
            self.dataset.dataset,
            self.dataset.sequence,
            self.name,
        )
        self.checkpoint_dir = osp.join(self.work_dir, "checkpoints")
        self.summary_dir = osp.join(self.work_dir, "summaries")
        self.summary_writer = tensorboard.SummaryWriter(self.summary_dir)

        self.model, self.state = self.build_model()
        logging.info("* Loading checkpoint.")
        self.state = checkpoints.restore_checkpoint(
            self.checkpoint_path or self.checkpoint_dir,
            self.state,
            self.checkpoint_step,
        )
        self.init_step = self.state.optimizer.state.step + 1

        self.tasks = self.build_tasks()

    def build_dataset(self) -> Dataset:
        log_str = (
            "* Creating dataset."
            if self.training
            else "* Creating dummy dataset."
        )
        create_fn = (
            self.dataset_cls.create
            if self.training
            else self.dataset_cls.create_dummy
        )
        logging.info(log_str)
        dataset = create_fn(training=self.training)
        return dataset

    def build_model(self) -> Tuple[nn.Module, struct.TrainState]:
        logging.info("* Creating model.")
        # Match the randomness in HyperNeRF repo.
        self.key, key = random.split(random.PRNGKey(0))
        cameras = common.parallel_map(
            self.dataset.parser.load_camera,
            self.dataset.time_ids,
            self.dataset.camera_ids,
        )
        cameras_dict = common.tree_collate(
            [
                {
                    "intrin": c.intrin,
                    "extrin": c.extrin,
                    "radial_distortion": c.radial_distortion,
                    "tangential_distortion": c.tangential_distortion,
                    "image_size": c.image_size,
                }
                for c in cameras
            ],
            collate_fn=lambda *x: jnp.array(x),
        )
        model, variables = self.model_cls.create(
            key=key,
            embeddings_dict=self.dataset.embeddings_dict,
            cameras_dict=cameras_dict,
            near=self.dataset.near,
            far=self.dataset.far,
        )
        optimizer = optim.Adam(0).create(variables)
        state = struct.TrainState(optimizer=optimizer)
        return model, state

    def build_tasks(self) -> Tasks:
        return Tasks(self)

    def launch(self):
        raise NotImplementedError
