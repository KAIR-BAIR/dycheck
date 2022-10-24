#!/usr/bin/env python3
#
# File   : cross_view.py
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

import itertools
import os.path as osp
from typing import Optional, Sequence, Union

import gin
import jax
import numpy as np
from absl import logging

from dycheck.utils import common, image, io, struct, types

from . import base
from .functional import get_prender_image


@gin.configurable(denylist=["engine"])
class CrossView(base.Task):
    """Render cross view for qualitative results.

    This task is particular useful when no multi-view validation is available.
    """

    def __init__(
        self,
        engine: types.EngineType,
        split: Union[Sequence[str], str] = gin.REQUIRED,
        *,
        interval: Optional[int] = None,
        num_steps: int = 3,
        force: bool = False,
    ):
        super().__init__(engine, interval=interval)
        if isinstance(split, str):
            split = [split]
        self.split = split
        self.num_steps = num_steps
        self.force = force

    @property
    def eligible(self):
        # Only perform this task when there is no multi-view validation.
        return not self.engine.dataset.has_novel_view or self.force

    def start(self):
        engine = self.engine

        if not hasattr(engine, "renders_dir"):
            engine.renders_dir = osp.join(engine.work_dir, "renders")
        self.render_dir = osp.join(engine.renders_dir, "cross_view")
        if not hasattr(engine, "eval_datasets"):
            engine.eval_datasets = dict()
        self.cache = dict()
        for split in self.split:
            if split not in engine.eval_datasets:
                engine.eval_datasets[split] = engine.dataset_cls.create(
                    split=split,
                    training=False,
                )
            dataset = engine.eval_datasets[split]
            rays = common.tree_collate(
                [
                    batch["rays"]
                    for batch in common.strided_subset(dataset, self.num_steps)
                ]
            )
            self.cache[split] = {
                "rays": [
                    jax.tree_map(lambda v: v[i], rays)
                    for i in range(self.num_steps)
                ],
                "metadata": [
                    jax.tree_map(lambda v: v[i], rays.metadata)
                    for i in range(self.num_steps)
                ],
            }
        self.prender_image = get_prender_image(engine.model)

    def every_n_steps(self):
        engine = self.engine

        for split in self.split:
            combined_img = self._render_cross_view_grid(
                self.cache[split]["rays"],
                self.cache[split]["metadata"],
                desc=f"* Rendering single cross view ({split})",
            )
            logging.info(f"* Single cross view rendered ({split}).")
            io.dump(
                osp.join(
                    self.render_dir,
                    split,
                    "checkpoints",
                    f"{engine.step:07d}.png",
                ),
                combined_img,
            )
            engine.summary_writer.image(
                f"cross_view/{split}",
                combined_img,
                engine.step,
            )

    def finalize(self):
        for split in self.split:
            combined_img = self._render_cross_view_grid(
                self.cache[split]["rays"],
                self.cache[split]["metadata"],
                desc=f"* Rendering single cross view ({split})",
            )
            logging.info(f"* Single cross view finalized ({split}).")
            io.dump(
                osp.join(
                    self.render_dir,
                    split,
                    f"num_steps_{self.num_steps:02d}.png",
                ),
                combined_img,
            )

    def _render_cross_view_grid(
        self,
        rays: struct.Rays,
        metadata: struct.Metadata,
        desc: str,
    ):
        engine = self.engine

        H, W = rays[0].origins.shape[:2]
        pbar = common.tqdm(
            itertools.product(rays, metadata),
            total=self.num_steps**2,
            desc=desc,
        )
        combined_imgs = []
        for rays, metadata in pbar:
            rays = rays._replace(metadata=metadata)
            rendered = self.prender_image(
                engine.pstate.optimizer.target,
                rays,
                key=engine.key,
                show_pbar=False,
            )
            pred_rgb = image.to_quantized_float32(rendered["rgb"])
            combined_imgs.append(pred_rgb)
        combined_imgs = np.array(combined_imgs).reshape(
            self.num_steps, self.num_steps, H, W, 3
        )
        combined_imgs = combined_imgs.transpose(0, 2, 1, 3, 4).reshape(
            self.num_steps * H, self.num_steps * W, 3
        )
        return combined_imgs
