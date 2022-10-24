#!/usr/bin/env python3
#
# File   : novel_view.py
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
from collections import OrderedDict, defaultdict
from typing import Optional, Sequence, Union

import gin
import numpy as np
from absl import logging

from dycheck.utils import common, image, io, types

from .. import metrics
from . import base, utils
from .functional import get_prender_image


@gin.configurable(denylist=["engine"])
class NovelView(base.Task):
    """Render novel view for all splits and compute metrics.

    Note that for all rgb predictions, we use the quantized version for
    computing metrics such that the results are consistent when loading saved
    images afterwards.
    """

    def __init__(
        self,
        engine: types.EngineType,
        split: Union[Sequence[str], str] = gin.REQUIRED,
        *,
        interval: Optional[int] = None,
    ):
        super().__init__(engine, interval=interval)
        if isinstance(split, str):
            split = [split]
        self.split = split

        self._step_stats = defaultdict(int)

    @property
    def eligible(self):
        return self.engine.dataset.has_novel_view

    def start(self):
        engine = self.engine

        if not hasattr(engine, "renders_dir"):
            engine.renders_dir = osp.join(engine.work_dir, "renders")
        self.render_dir = osp.join(engine.renders_dir, "novel_view")
        if not hasattr(engine, "eval_datasets"):
            engine.eval_datasets = dict()
        for split in self.split:
            if split not in engine.eval_datasets:
                engine.eval_datasets[split] = engine.dataset_cls.create(
                    split=split,
                    training=False,
                )
        self.prender_image = get_prender_image(engine.model)

        self.compute_lpips = metrics.get_compute_lpips()

    def every_n_steps(self):
        engine = self.engine

        for split in self.split:
            dataset = engine.eval_datasets[split]
            batch = dataset[self._step_stats[split]]
            rendered = self.prender_image(
                engine.pstate.optimizer.target,
                batch["rays"],
                key=engine.key,
                desc=f"* Rendering single novel view ({split})",
            )
            rgb = image.to_quantized_float32(batch["rgb"])
            mask = image.to_quantized_float32(batch["mask"])
            pred_rgb = image.to_quantized_float32(rendered["rgb"])
            metrics_dict = {
                "psnr": metrics.compute_psnr(rgb, pred_rgb, mask).item(),
                "ssim": metrics.compute_ssim(rgb, pred_rgb, mask).item(),
                "lpips": self.compute_lpips(rgb, pred_rgb, mask).item(),
            }
            combined_imgs = [rgb, pred_rgb]
            if "covisible" in batch:
                covisible = image.to_quantized_float32(batch["covisible"])
                metrics_dict.update(
                    **{
                        "mpsnr": metrics.compute_psnr(
                            rgb, pred_rgb, covisible
                        ).item(),
                        "mssim": metrics.compute_ssim(
                            rgb, pred_rgb, covisible
                        ).item(),
                        "mlpips": self.compute_lpips(
                            rgb, pred_rgb, covisible
                        ).item(),
                    }
                )
                # Mask out the non-covisible region by white color.
                covisible_pred_rgb = (
                    covisible * pred_rgb + (1 - covisible) * (1 + pred_rgb) / 2
                )
                combined_imgs.append(covisible_pred_rgb)
            logging.info(
                (
                    f"* Single novel view metrics ({split}):\n"
                    f"{utils.format_dict(metrics_dict)}"
                )
            )
            combined_imgs = np.concatenate(combined_imgs, axis=1)
            io.dump(
                osp.join(
                    self.render_dir,
                    split,
                    "checkpoints",
                    f"{engine.step:07d}.png",
                ),
                combined_imgs,
            )
            engine.summary_writer.image(
                f"novel_view/{split}",
                combined_imgs,
                engine.step,
            )
            for k, v in metrics_dict.items():
                engine.summary_writer.scalar(
                    f"novel_view/{split}/{k}", v, engine.step
                )

            self._step_stats[split] += 1
            self._step_stats[split] %= len(dataset)

    def finalize(self):
        engine = self.engine

        for split in self.split:
            # Recreate the dataset such that the iterator is reset.
            dataset = engine.dataset_cls.create(
                split=split,
                training=False,
            )
            metrics_dicts = []
            pbar = common.tqdm(
                range(len(dataset)),
                desc=f"* Rendering novel views ({split})",
            )
            for i, batch in zip(pbar, dataset):
                frame_name = dataset.frame_names[i]
                rendered = self.prender_image(
                    engine.pstate.optimizer.target,
                    batch["rays"],
                    key=engine.key,
                    show_pbar=False,
                )
                rgb = image.to_quantized_float32(batch["rgb"])
                mask = image.to_quantized_float32(batch["mask"])
                pred_rgb = image.to_quantized_float32(rendered["rgb"])
                metrics_dict = OrderedDict(
                    {
                        "frame_name": frame_name,
                        "psnr": metrics.compute_psnr(
                            rgb, pred_rgb, mask
                        ).item(),
                        "ssim": metrics.compute_ssim(
                            rgb, pred_rgb, mask
                        ).item(),
                        "lpips": self.compute_lpips(
                            rgb, pred_rgb, mask
                        ).item(),
                    }
                )
                combined_imgs = [rgb, pred_rgb]
                if "covisible" in batch:
                    covisible = image.to_quantized_float32(batch["covisible"])
                    metrics_dict.update(
                        **{
                            "mpsnr": metrics.compute_psnr(
                                rgb, pred_rgb, covisible
                            ).item(),
                            "mssim": metrics.compute_ssim(
                                rgb, pred_rgb, covisible
                            ).item(),
                            "mlpips": self.compute_lpips(
                                rgb, pred_rgb, covisible
                            ).item(),
                        }
                    )
                    # Mask out the non-covisible region by white color.
                    covisible_pred_rgb = (
                        covisible * pred_rgb
                        + (1 - covisible) * (1 + pred_rgb) / 2
                    )
                    combined_imgs.append(covisible_pred_rgb)
                pbar.set_description(
                    f"* Rendering novel view ({split}), "
                    + ", ".join(
                        f"{k}: {v:.3f}"
                        for k, v in metrics_dict.items()
                        if k != "frame_name"
                    )
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + ".png"),
                    np.concatenate(combined_imgs, axis=1),
                )
                # Skip logging to tensorboard bc it's a lot of images.
                metrics_dicts.append(metrics_dict)
            metrics_dict = common.tree_collate(metrics_dicts)
            io.dump(
                osp.join(self.render_dir, split, "metrics_dict.npz"),
                **metrics_dict,
            )
            mean_metrics_dict = {
                k: float(v.mean())
                for k, v in metrics_dict.items()
                if k != "frame_name"
            }
            io.dump(
                osp.join(self.render_dir, split, "mean_metrics_dict.json"),
                mean_metrics_dict,
                sort_keys=False,
            )
            logging.info(
                (
                    f"* Mean novel view metrics ({split}):\n"
                    f"{utils.format_dict(mean_metrics_dict)}"
                )
            )
