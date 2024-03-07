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
import pickle as pkl
from collections import OrderedDict
from typing import Optional, Sequence, Union

import gin
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from tqdm import tqdm

from dycheck import datasets
from dycheck.geometry.camera import Camera
from dycheck.utils import common, image, io, struct, types, visuals

from .. import metrics
from . import base, utils
from .functional import get_prender_image, get_pwarp_pixels

NUM_KEYPOINT_FRAMES = 10


@gin.configurable(denylist=["engine"])
class KeypointTransferForFigure(base.Task):
    """Transfer keypoints across frames for all splits and compute metrics."""

    def __init__(
        self,
        engine: types.EngineType,
        *,
        dump_dir: str,
    ):
        super().__init__(engine, interval=None)
        self.dump_dir = dump_dir

    @property
    def eligible(self):
        return True

    def start(self):
        engine = self.engine

        if not hasattr(engine, "renders_dir"):
            engine.renders_dir = osp.join(engine.work_dir, "renders")
        self.results_dir = osp.join(
            engine.work_dir, "results", "kp_transfer_for_figure"
        )
        with open(osp.join(self.dump_dir, "track_dicts.pkl"), "rb") as f:
            self.track_dicts = pkl.load(f)
        self.prender_image = get_prender_image(engine.model)
        self.pwarp_pixels = get_pwarp_pixels(engine.model)

    def finalize(self):
        engine = self.engine

        dataset = engine.dataset_cls.create(split="train", training=False)
        cameras = dataset.cameras

        def normalize(x, axis):
            return x / np.linalg.norm(x, axis=axis, keepdims=True)

        train_c2ws = np.linalg.inv(np.stack([c.extrin for c in cameras], 0))
        c2ws = train_c2ws
        center = c2ws[:, :3, -1].mean(0)
        z = normalize(c2ws[:, :3, 2].mean(0), 0)  # (3)
        y_ = c2ws[:, :3, 1].mean(0)  # (3)
        x = normalize(np.cross(y_, z), 0)  # (3)
        y = np.cross(z, x)  # (3)
        c2w = np.eye(4)
        c2w[:3] = np.stack([x, y, z, center], 1)  # (3, 4)
        w2c = np.linalg.inv(c2w)
        K = cameras[0].intrin
        camera = cameras[0].copy()
        camera.orientation = w2c[:3, :3]
        camera.position = c2w[:3, -1]
        camera.focal_length = K[0, 0]
        camera.principal_point = K[:2, -1]
        camera.pixel_aspect_ratio = K[1, 1] / K[0, 0]

        pbar = common.tqdm(
            self.track_dicts,
            desc=f"* Transferring keypoints for figure",
        )
        result_dicts = []
        for track_dict in pbar:
            frame_name = track_dict["frame_name"]
            query_pixels = track_dict["query_pixels"]
            t = track_dict["t"]
            target_ts = track_dict["target_ts"]
            rays = camera.pixels_to_rays(camera.get_pixels())
            rays = jax.tree_map(lambda x: jnp.array(x), rays)
            rays = rays._replace(
                metadata=struct.Metadata(
                    time=jnp.full(
                        (*camera.image_shape, 1), t, dtype=np.uint32
                    ),
                    camera=jnp.zeros(
                        (*camera.image_shape, 1), dtype=np.uint32
                    ),
                )
            )
            rendered = self.prender_image(
                engine.pstate.optimizer.target,
                rays,
                key=engine.key,
                show_pbar=True,
                pbar_kwargs={"leave": False, "position": 1},
            )
            pred_rgb = rendered["rgb"]
            io.dump(osp.join(self.results_dir, f"{frame_name}.png"), pred_rgb)
            query_pixels = jnp.array(track_dict["query_pixels"])
            tracks_2d = []
            for target_t in tqdm(
                track_dict["target_ts"], leave=False, position=1
            ):
                metadata = struct.Metadata(
                    time=np.full_like(
                        query_pixels[..., :1],
                        t,
                        dtype=np.uint32,
                    ),
                    time_to=np.full_like(
                        query_pixels[..., :1],
                        target_t,
                        dtype=np.uint32,
                    ),
                    camera=np.zeros_like(
                        query_pixels[..., :1],
                        dtype=np.uint32,
                    ),
                )
                tracks_2d.append(
                    self.pwarp_pixels(
                        engine.pstate.optimizer.target,
                        query_pixels,
                        metadata,
                        camera,
                        camera,
                        key=engine.key,
                        show_pbar=False,
                    )["warped_pixels"]
                )
            tracks_2d = jnp.stack(tracks_2d, 0)
            result_dicts.append(
                {
                    "tracks_2d": np.array(tracks_2d),
                    "img": (np.array(pred_rgb) * 255.0).astype(np.uint8),
                }
            )
        with open(osp.join(self.results_dir, "result_dicts.pkl"), "wb") as f:
            pkl.dump(result_dicts, f)
