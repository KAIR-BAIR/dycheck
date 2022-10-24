#!/usr/bin/env python3
#
# File   : kp_transfer.py
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
from collections import OrderedDict
from typing import Optional, Sequence, Union

import gin
import jax
import numpy as np
from absl import logging

from dycheck import datasets
from dycheck.utils import common, image, io, struct, types, visuals

from .. import metrics
from . import base, utils
from .functional import get_pwarp_pixels

NUM_KEYPOINT_FRAMES = 10


def get_kp_dataset(dataset: datasets.Dataset):
    frame_names, time_ids, camera_ids = jax.tree_map(
        lambda s: common.strided_subset(s, NUM_KEYPOINT_FRAMES),
        [
            dataset.frame_names,
            dataset.time_ids,
            dataset.camera_ids,
        ],
    )
    rgbs = common.parallel_map(
        lambda t, c: dataset.parser.load_rgba(t, c)[..., :3],
        time_ids,
        camera_ids,
    )
    cameras = common.parallel_map(
        dataset.parser.load_camera,
        time_ids,
        camera_ids,
    )
    keypoints = common.parallel_map(
        lambda t, c: dataset.parser.load_keypoints(t, c, dataset.split),
        time_ids,
        camera_ids,
    )

    def data_iter():
        for frame_name, time_id, camera_id, rgb, camera, kps in zip(
            frame_names, time_ids, camera_ids, rgbs, cameras, keypoints
        ):
            batch = {
                "frame_name": frame_name,
                "time_id": time_id,
                "camera_id": camera_id,
                "rgb": rgb,
                "camera": camera,
                "keypoints": kps,
            }
            yield batch

    return itertools.permutations(data_iter(), 2)


def visualize_kpt(
    kps: np.ndarray,
    kps_to: np.ndarray,
    pred_kps_to: np.ndarray,
    rgb: np.ndarray,
    rgb_to: np.ndarray,
    corrects: np.ndarray,
    *,
    skeleton: visuals.Skeleton,
    **kwargs,
):
    pred_kp_rgbs = np.array(
        [[0, 255, 0] if correct else [255, 0, 0] for correct in corrects],
        np.uint8,
    )
    return np.concatenate(
        common.parallel_map(
            lambda k, img, r: visuals.visualize_kps(
                k, img, skeleton=skeleton, rgbs=r, **kwargs
            ),
            [kps, kps_to, pred_kps_to],
            [rgb, rgb_to, rgb_to],
            [
                np.array([0, 255, 0], np.uint8),
                np.array([0, 255, 0], np.uint8),
                pred_kp_rgbs,
            ],
        ),
        axis=1,
    )


@gin.configurable(denylist=["engine"])
class KeypointTransfer(base.Task):
    """Transfer keypoints across frames for all splits and compute metrics."""

    def __init__(
        self,
        engine: types.EngineType,
        split: Union[Sequence[str], str] = gin.REQUIRED,
        *,
        interval: Optional[int] = None,
        ratio: float = 0.05,
        kp_radius: int = 6,
        bone_thickness: int = 3,
    ):
        super().__init__(engine, interval=interval)
        if isinstance(split, str):
            split = [split]
        self.split = split
        self.ratio = ratio
        self.kp_radius = kp_radius
        self.bone_thickness = bone_thickness

    @property
    def eligible(self):
        return self.engine.dataset.has_keypoints

    def start(self):
        engine = self.engine

        if not hasattr(engine, "renders_dir"):
            engine.renders_dir = osp.join(engine.work_dir, "renders")
        self.render_dir = osp.join(engine.renders_dir, "kp_transfer")
        if not hasattr(engine, "skeletons"):
            engine.skeletons = dict()
        # Keypoint dataset is private to this task.
        self.kp_datasets = dict()
        for split in self.split:
            if split not in self.kp_datasets or split not in engine.skeletons:
                dataset = engine.dataset_cls.create(
                    split=split,
                    training=False,
                )
                self.kp_datasets[split] = get_kp_dataset(dataset)
                engine.skeletons[split] = dataset.parser.load_skeleton(split)
        self.pwarp_pixels = get_pwarp_pixels(engine.model)

    def every_n_steps(self):
        engine = self.engine

        for split in self.split:
            dataset = self.kp_datasets[split]
            skeleton = engine.skeletons[split]
            batch, batch_to = next(dataset)
            keypoints, keypoints_to = batch["keypoints"], batch_to["keypoints"]
            camera, camera_to = batch["camera"], batch_to["camera"]
            # Get common keypoints.
            mask = (keypoints[..., -1] == 1) & (keypoints_to[..., -1] == 1)
            if mask.sum() == 0:
                logging.info(
                    f"* Transferring single keypoints pair failed ({split}): "
                    f"no common keypoints."
                )
                continue
            common_keypoints, common_keypoints_to = (
                keypoints[mask][..., :2],
                keypoints_to[mask][..., :2],
            )
            metadata = struct.Metadata(
                time=np.full_like(
                    common_keypoints[..., :1],
                    batch["time_id"],
                    dtype=np.uint32,
                ),
                time_to=np.full_like(
                    common_keypoints[..., :1],
                    batch_to["time_id"],
                    dtype=np.uint32,
                ),
                camera=np.full_like(
                    common_keypoints[..., :1],
                    batch["camera_id"],
                    dtype=np.uint32,
                ),
            )
            warped = self.pwarp_pixels(
                engine.pstate.optimizer.target,
                common_keypoints,
                metadata,
                camera,
                camera_to,
                key=engine.key,
                desc=f"* Transferring single keypoints pair ({split})",
            )
            pred_common_keypoints_to = warped["warped_pixels"]
            common_corrects = metrics.compute_pck(
                common_keypoints_to,
                pred_common_keypoints_to,
                camera.image_size,
                self.ratio,
                reduce=None,
            )
            metrics_dict = {
                f"pck@{self.ratio:.3f}": common_corrects.mean().item(),
            }
            logging.info(
                (
                    f"* Single keypoint transfer metrics ({split}):\n"
                    f"{utils.format_dict(metrics_dict)}"
                )
            )
            rgb = image.to_float32(batch["rgb"])
            rgb_to = image.to_float32(batch_to["rgb"])
            pred_keypoints_to = np.zeros_like(keypoints)
            pred_keypoints_to[mask] = np.concatenate(
                [
                    pred_common_keypoints_to,
                    np.ones_like(pred_common_keypoints_to[..., :1]),
                ],
                axis=-1,
            )
            corrects = np.zeros_like(keypoints[..., 0], dtype=bool)
            corrects[mask] = common_corrects
            kp_visual = visualize_kpt(
                np.where(mask[..., None], keypoints, 0),
                np.where(mask[..., None], keypoints_to, 0),
                pred_keypoints_to,
                rgb,
                rgb_to,
                corrects,
                skeleton=skeleton,
                kp_radius=self.kp_radius,
                bone_thickness=self.bone_thickness,
            )
            io.dump(
                osp.join(
                    self.render_dir,
                    split,
                    "checkpoints",
                    f"{engine.step:07d}.png",
                ),
                kp_visual,
            )
            engine.summary_writer.image(
                f"kp_transfer/{split}",
                kp_visual,
                engine.step,
            )
            for k, v in metrics_dict.items():
                engine.summary_writer.scalar(
                    f"kp_transfer/{split}/{k}", v, engine.step
                )

    def finalize(self):
        engine = self.engine

        for split in self.split:
            # Recreate the dataset such that the iterator is reset.
            dataset = get_kp_dataset(
                engine.dataset_cls.create(
                    split=split,
                    training=False,
                )
            )
            skeleton = engine.skeletons[split]
            metrics_dicts = []
            pbar = common.tqdm(
                dataset,
                desc=f"* Transferring keypoints pairs ({split})",
                total=NUM_KEYPOINT_FRAMES * (NUM_KEYPOINT_FRAMES - 1),
            )
            for batch, batch_to in pbar:
                frame_name, frame_name_to = (
                    batch["frame_name"],
                    batch_to["frame_name"],
                )
                keypoints, keypoints_to = (
                    batch["keypoints"],
                    batch_to["keypoints"],
                )
                camera, camera_to = batch["camera"], batch_to["camera"]
                # Get common keypoints.
                mask = (keypoints[..., -1] == 1) & (keypoints_to[..., -1] == 1)
                if mask.sum() == 0:
                    continue
                common_keypoints, common_keypoints_to = (
                    keypoints[mask][..., :2],
                    keypoints_to[mask][..., :2],
                )
                metadata = struct.Metadata(
                    time=np.full_like(
                        common_keypoints[..., :1],
                        batch["time_id"],
                        dtype=np.uint32,
                    ),
                    time_to=np.full_like(
                        common_keypoints[..., :1],
                        batch_to["time_id"],
                        dtype=np.uint32,
                    ),
                    camera=np.full_like(
                        common_keypoints[..., :1],
                        batch["camera_id"],
                        dtype=np.uint32,
                    ),
                )
                warped = self.pwarp_pixels(
                    engine.pstate.optimizer.target,
                    common_keypoints,
                    metadata,
                    camera,
                    camera_to,
                    key=engine.key,
                    show_pbar=False,
                )
                pred_common_keypoints_to = warped["warped_pixels"]
                common_corrects = metrics.compute_pck(
                    common_keypoints_to,
                    pred_common_keypoints_to,
                    camera.image_size,
                    self.ratio,
                    reduce=None,
                )
                metrics_dict = OrderedDict(
                    {
                        "frame_name": frame_name,
                        "frame_name_to": frame_name_to,
                        f"pck@{self.ratio:.3f}": common_corrects.mean().item(),
                    }
                )
                pbar.set_description(
                    f"* Transferring keypoints pairs ({split}), "
                    + ", ".join(
                        f"{k}: {v:.3f}"
                        for k, v in metrics_dict.items()
                        if k not in ["frame_name", "frame_name_to"]
                    )
                )
                rgb = image.to_float32(batch["rgb"])
                rgb_to = image.to_float32(batch_to["rgb"])
                pred_keypoints_to = np.zeros_like(keypoints)
                pred_keypoints_to[mask] = np.concatenate(
                    [
                        pred_common_keypoints_to,
                        np.ones_like(pred_common_keypoints_to[..., :1]),
                    ],
                    axis=-1,
                )
                corrects = np.zeros_like(keypoints[..., 0], dtype=bool)
                corrects[mask] = common_corrects
                metrics_dict["pred_keypoints_to"] = pred_keypoints_to
                kp_visual = visualize_kpt(
                    np.where(mask[..., None], keypoints, 0),
                    np.where(mask[..., None], keypoints_to, 0),
                    pred_keypoints_to,
                    rgb,
                    rgb_to,
                    corrects,
                    skeleton=skeleton,
                    kp_radius=self.kp_radius,
                    bone_thickness=self.bone_thickness,
                )
                io.dump(
                    osp.join(
                        self.render_dir,
                        split,
                        f"{frame_name}-{frame_name_to}.png",
                    ),
                    kp_visual,
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
                if k
                not in ["frame_name", "frame_name_to", "pred_keypoints_to"]
            }
            io.dump(
                osp.join(self.render_dir, split, "mean_metrics_dict.json"),
                mean_metrics_dict,
                sort_keys=False,
            )
            logging.info(
                (
                    f"* Mean keypoint transfer metrics ({split}):\n"
                    f"{utils.format_dict(mean_metrics_dict)}"
                )
            )
