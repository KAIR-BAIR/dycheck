#!/usr/bin/env python3
#
# File   : process_covisible.py
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

import dataclasses
import functools
import os.path as osp
from typing import Callable, Optional, Sequence

import gin
import jax
import numpy as np
import torch
from absl import app, flags, logging

from dycheck import core
from dycheck.datasets import Parser
from dycheck.processors import raft
from dycheck.utils import common, image, io, types

flags.DEFINE_multi_string(
    "gin_configs", None, "Gin config files.", required=True
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
FLAGS = flags.FLAGS


@gin.configurable(module="process_covisible")
@dataclasses.dataclass
class Config(object):
    parser_cls: Callable[..., Parser] = gin.REQUIRED
    splits: Sequence[str] = gin.REQUIRED
    chunk: int = gin.REQUIRED
    num_min_frames: int = 5
    min_frame_ratio: float = 0.1


def load_split_and_support(parser: Parser, split: str):
    frame_names, time_ids, camera_ids = parser.load_split(split)
    support_frame_names, support_time_ids, support_camera_ids = (
        parser.frame_names,
        parser.time_ids,
        parser.camera_ids,
    )
    mask = ~np.isin(support_frame_names, frame_names)
    support_frame_names, support_time_ids, support_camera_ids = jax.tree_map(
        lambda x: x[mask],
        (support_frame_names, support_time_ids, support_camera_ids),
    )
    return (frame_names, time_ids, camera_ids), (
        support_frame_names,
        support_time_ids,
        support_camera_ids,
    )


class RAFTDensePairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        imgs: np.ndarray,
        support_imgs: np.ndarray,
        masks: np.ndarray,
        support_masks: np.ndarray,
    ):
        self.imgs = torch.from_numpy(imgs.astype(np.float32)).permute(
            0, 3, 1, 2
        )
        self.support_imgs = torch.from_numpy(
            support_imgs.astype(np.float32)
        ).permute(0, 3, 1, 2)
        self.masks = image.to_float32(masks)
        self.support_masks = image.to_float32(support_masks)

        # Number of base and support images.
        self.B, self.S = len(self.imgs), len(self.support_imgs)

    def __getitem__(self, index: int):
        bi, si = divmod(index, self.S)
        return (
            self.imgs[bi],
            self.support_imgs[si],
            self.masks[bi],
            self.support_masks[si],
        )

    def __len__(self):
        return self.B * self.S


@torch.inference_mode()
def dump_covisible(
    covisible_dir: types.PathType,
    frame_names: Sequence[str],
    imgs: np.ndarray,
    support_imgs: np.ndarray,
    masks: np.ndarray,
    support_masks: np.ndarray,
    *,
    chunk: int,
    desc: Optional[str] = None,
    num_min_frames: int = 5,
    min_frame_ratio: float = 0.1,
):
    dataset = RAFTDensePairDataset(imgs, support_imgs, masks, support_masks)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=chunk,
        shuffle=False,
    )
    model = torch.nn.DataParallel(raft.get_raft()).to("cuda").eval()
    padder = raft.InputPadder(dataset.imgs.shape)

    cache = []
    # Index of the base image to dump.
    bi = 0
    for img0, img1, mask0, mask1 in common.tqdm(data_loader, desc=desc):
        img0, img1 = jax.tree_map(lambda x: x.to("cuda"), (img0, img1))
        img0, img1 = padder.pad(img0, img1)
        mask0, mask1 = jax.tree_map(lambda x: x.numpy(), (mask0, mask1))

        def _compute_flow(x0, x1):
            flow = model(x0, x1, iters=20, test_mode=True)[1]
            flow = padder.unpad(flow).permute(0, 2, 3, 1).cpu().numpy()
            return flow

        flow_fw = _compute_flow(img0, img1)
        flow_bw = _compute_flow(img1, img0)
        masked_flow_fw = np.where(mask0 == 1, flow_fw, np.inf)
        masked_flow_bw = np.where(mask1 == 1, flow_bw, np.inf)
        # list of C.
        cache += list(
            common.parallel_map(
                raft.compute_occ_brox, masked_flow_fw, masked_flow_bw
            )
            + (1 - mask0)
        )
        while len(cache) >= dataset.S:
            support_occs = np.array(cache[: dataset.S], np.float32)
            cache = cache[dataset.S :]
            covisible = 1 - (
                (1 - support_occs).sum(axis=0)
                <= max(num_min_frames, int(min_frame_ratio * dataset.S))
            ).astype(np.float32)
            io.dump(
                osp.join(covisible_dir, frame_names[bi] + ".png"), covisible
            )
            bi += 1


def main(_):
    logging.info(f"*** Loading Gin configs from: {FLAGS.gin_configs}.")
    core.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs,
        bindings=FLAGS.gin_bindings,
        skip_unknown=True,
        master=False,
    )

    config_str = gin.config_str()
    logging.info(f"*** Configuration:\n{config_str}")

    config = Config()

    logging.info("*** Starting processing covisible masks.")
    parser = config.parser_cls()
    assert not getattr(
        parser, "use_undistort", False
    ), "Covisiblity undistortion should be done at parsing runtime."

    for split in config.splits:
        covisible_dir = osp.join(
            parser.data_dir, "covisible", f"{parser.factor}x", split
        )
        if split == "val_common":
            frame_names, time_ids, camera_ids = parser.load_split("val_common")
            covisible_intls = image.to_float32(
                np.array(
                    common.parallel_map(
                        functools.partial(
                            parser.load_covisible, split="val_intl"
                        ),
                        time_ids,
                        camera_ids,
                    )
                )
            )
            covisible_monos = image.to_float32(
                np.array(
                    common.parallel_map(
                        functools.partial(
                            parser.load_covisible, split="val_mono"
                        ),
                        time_ids,
                        camera_ids,
                    )
                )
            )
            # Covisible in both splits
            covisibles = covisible_intls * covisible_monos
            common.parallel_map(
                io.dump,
                [
                    osp.join(covisible_dir, name + ".png")
                    for name in frame_names
                ],
                covisibles,
                show_pbar=True,
                desc=f"* Dump {split}",
            )
        else:
            (frame_names, time_ids, camera_ids), (
                _,
                support_time_ids,
                support_camera_ids,
            ) = load_split_and_support(parser, split)
            imgs, masks = np.split(
                np.array(
                    common.parallel_map(parser.load_rgba, time_ids, camera_ids)
                ),
                (3,),
                axis=-1,
            )
            support_imgs, support_masks = np.split(
                np.array(
                    common.parallel_map(
                        parser.load_rgba, support_time_ids, support_camera_ids
                    )
                ),
                (3,),
                axis=-1,
            )
            dump_covisible(
                covisible_dir,
                frame_names,
                imgs,
                support_imgs,
                masks,
                support_masks,
                chunk=config.chunk,
                desc=f"* Dumping {split}",
                num_min_frames=config.num_min_frames,
                min_frame_ratio=config.min_frame_ratio,
            )


if __name__ == "__main__":
    app.run(main)
