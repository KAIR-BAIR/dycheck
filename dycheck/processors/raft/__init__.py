#!/usr/bin/env python3
#
# File   : __init__.py
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
import os.path as osp
import shutil
from typing import Callable, NamedTuple, Optional

import cv2
import jax
import numpy as np
import torch
from addict import Dict
from torch import nn

from dycheck.utils import common, image, path_ops

from ._impl.raft import RAFT  # type: ignore
from ._impl.utils.utils import InputPadder  # type: ignore


class RAFTFlow(NamedTuple):
    flow_fw: np.ndarray
    flow_bw: np.ndarray
    occ_fw: np.ndarray
    occ_bw: np.ndarray


def flow_to_warp(flow: np.ndarray) -> np.ndarray:
    """Compute the warp from the flow field.

    Args:
        flow (np.ndarray): an optical flow image of shape (H, W, 2).

    Returns:
        warp (np.ndarray): the endpoints representation of shape (H, W, 2).
    """
    H, W = flow.shape[:2]
    x, y = np.meshgrid(
        np.arange(W, dtype=flow.dtype),
        np.arange(H, dtype=flow.dtype),
        indexing="xy",
    )
    grid = np.stack([x, y], axis=-1)

    warp = grid + flow
    return warp


def compute_occ_brox(flow_fw: np.ndarray, flow_bw: np.ndarray):
    """Compute an occlusion mask based on a forward-backward check.

    High accuracy optical flow estimation based on a theory for warping.
        Brox et al., ECCV 2004.
        https://link.springer.com/chapter/10.1007/978-3-540-24673-2_3

    Args:
        flow_fw (np.ndarray): forward flow field of shape (H, W, 2).
        flow_bw (np.ndarray) : backward flow field of shape (H, W, 2).

    Returns:
        occ (np.ndarray): hard occlusion mask of shape (H, W, 1) where 1
            represents locations ``to-be-occluded''.
    """
    # Resampled backward flow at forward flow locations.
    warp = flow_to_warp(flow_fw)

    backward_flow_resampled = cv2.remap(
        flow_bw, warp[..., 0], warp[..., 1], cv2.INTER_LINEAR
    )
    # Compute occlusions based on forward-backward consistency.
    fb_sq_diff = np.sum(
        (flow_fw + backward_flow_resampled) ** 2, axis=-1, keepdims=True
    )
    fb_sum_sq = np.sum(
        flow_fw**2 + backward_flow_resampled**2,
        axis=-1,
        keepdims=True,
    )
    return (fb_sq_diff > 0.01 * fb_sum_sq + 0.5).astype(np.float32)


def get_raft() -> torch.nn.Module:
    args = Dict(small=False, mixed_precision=False)
    model = nn.DataParallel(RAFT(args))
    model_dir = osp.join(torch.hub.get_dir(), "checkpoints/raft")
    if not osp.exists(model_dir):
        path_ops.mkdir(model_dir)
        model_zip = osp.join(model_dir, "models.zip")
        # From the original repo.
        torch.hub.download_url_to_file(
            "https://dl.dropbox.com/s/4j4z58wuv8o0mfz/models.zip", model_zip
        )
        shutil.unpack_archive(model_zip, model_dir)
    model.load_state_dict(
        torch.load(
            osp.join(model_dir, "models/raft-things.pth"), map_location="cpu"
        )
    )
    model = model.module
    return model


def get_compute_raft_flow(
    chunk: int = 0,
    show_pbar: bool = True,
    desc: Optional[str] = "* Compute RAFT flow",
) -> Callable[
    [np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
    RAFTFlow,
]:
    model = get_raft()
    if chunk == 0:
        fn = compute_raft_flow
    else:
        model = nn.DataParallel(model)
        fn = functools.partial(
            compute_chunk_raft_flow,
            chunk=chunk,
            show_pbar=show_pbar,
            desc=desc,
        )
    return functools.partial(fn, model.to("cuda").eval())


@torch.inference_mode()
def compute_raft_flow(
    model: RAFT,
    img0: np.ndarray,
    img1: np.ndarray,
    mask0: Optional[np.ndarray] = None,
    mask1: Optional[np.ndarray] = None,
) -> RAFTFlow:
    """Estimate flows with RAFT model for a pair of images.

    RAFT: Recurrent all-pairs field transforms for optical flow.
        Teed et al., ECCV 2020.
        https://arxiv.org/abs/2003.12039

    Note that RAFT model takes uint8 (or 255-scale float32) images as
    input.

    Args:
        img0 (np.ndarray): (H, W, 3), a source image in float32 or uint8
            RGB format. Note that it provides spatial support for flow
            estimation.
        img1 (np.ndarray): (H, W, 3), a destination image in float32 or
            uint8 RGB format.
        mask0 (Optional[np.ndarray]): (H, W, 1), a binary mask of the
            source image.
        mask1 (Optional[np.ndarray]): (H, W, 1), a binary mask of the
            destination image.

    Returns:
        RAFTFlow: a namedtuple of flow fields including flow and hard
            occlusion for both directions.
    """

    img0 = (
        torch.from_numpy(image.to_uint8(img0))
        .permute(2, 0, 1)
        .float()[None]
        .to("cuda")
    )
    img1 = (
        torch.from_numpy(image.to_uint8(img1))
        .permute(2, 0, 1)
        .float()[None]
        .to("cuda")
    )
    padder = InputPadder(img0.shape)
    img0, img1 = padder.pad(img0, img1)

    def _compute_flow(x0, x1):
        flow = model(x0, x1, iters=20, test_mode=True)[1]
        flow = padder.unpad(flow[0]).permute(1, 2, 0).cpu().numpy()
        return flow

    flow_fw = _compute_flow(img0, img1)
    flow_bw = _compute_flow(img1, img0)
    if mask0 is None:
        mask0 = np.ones_like(flow_fw[..., :1])
    else:
        mask0 = image.to_float32(mask0)
    if mask1 is None:
        mask1 = np.ones_like(flow_bw[..., :1])
    else:
        mask1 = image.to_float32(mask1)
    masked_flow_fw = np.where(mask0 == 1, flow_fw, np.inf)
    masked_flow_bw = np.where(mask1 == 1, flow_bw, np.inf)
    occ_fw = compute_occ_brox(masked_flow_fw, masked_flow_bw) + (1 - mask0)
    occ_bw = compute_occ_brox(masked_flow_bw, masked_flow_fw) + (1 - mask1)
    return RAFTFlow(flow_fw, flow_bw, occ_fw, occ_bw)


@torch.inference_mode()
def compute_chunk_raft_flow(
    model: RAFT,
    img0: np.ndarray,
    img1: np.ndarray,
    mask0: Optional[np.ndarray] = None,
    mask1: Optional[np.ndarray] = None,
    *,
    chunk: int = 16,
    show_pbar: bool = True,
    desc: Optional[str] = "* Compute RAFT flow",
) -> RAFTFlow:
    """Estimate flows with RAFT model for two sets of images in chunks/batches.

    RAFT: Recurrent all-pairs field transforms for optical flow.
        Teed et al., ECCV 2020.
        https://arxiv.org/abs/2003.12039

    Note that RAFT model takes uint8 (or 255-scale float32) images as
    input.
    This function perform reasonably well for small set of images (e.g. < 1k
    images). For more images, try using torch dataset and proper batching.

    Args:
        img0 (np.ndarray): (B, H, W, 3), source images in float32 or uint8 RGB
            format. Note that it provides spatial support for flow estimation.
        img1 (np.ndarray): (B, H, W, 3), destination images in float32 or uint8
            RGB format.
        mask0 (Optional[np.ndarray]): (B, H, W, 1), binary masks of the source
            image.
        mask1 (Optional[np.ndarray]): (B, H, W, 1), binary masks of the
            destination image.

    Returns:
        RAFTFlow: a namedtuple of flow fields including flow and hard
            occlusion for both directions.
    """

    B = img0.shape[0]
    assert chunk > 0
    num_devices = torch.cuda.device_count()

    if mask0 is None:
        mask0 = np.ones_like(img0[..., :1])
    else:
        mask0 = image.to_float32(mask0)
    if mask1 is None:
        mask1 = np.ones_like(img1[..., :1])
    else:
        mask1 = image.to_float32(mask1)

    img0 = (
        torch.from_numpy(image.to_uint8(img0))
        .permute(0, 3, 1, 2)
        .float()
        .to("cuda")
    )
    img1 = (
        torch.from_numpy(image.to_uint8(img1))
        .permute(0, 3, 1, 2)
        .float()
        .to("cuda")
    )
    padder = InputPadder(img0.shape)
    img0, img1 = padder.pad(img0, img1)

    def _compute_flow(x0, x1):
        flow = model(x0, x1, iters=20, test_mode=True)[1]
        flow = padder.unpad(flow).permute(0, 2, 3, 1).cpu().numpy()
        return flow

    results = []
    for i in (common.tqdm if show_pbar else lambda x, **_,: x)(
        range(0, B, chunk), desc=desc
    ):
        chunk_slice_fn = lambda x: x[i : i + chunk]
        chunk_img0, chunk_img1, chunk_mask0, chunk_mask1 = jax.tree_map(
            chunk_slice_fn, (img0, img1, mask0, mask1)
        )
        num_chunk_imgs = chunk_img0.shape[0]
        remainder = num_chunk_imgs % num_devices
        if remainder != 0:
            padding = num_devices - remainder
            chunk_pad_fn = lambda x: torch.cat(
                [x, x[-1:].repeat_interleave(padding, dim=0)], dim=0
            )
            chunk_img0, chunk_img1 = jax.tree_map(
                chunk_pad_fn, (chunk_img0, chunk_img1)
            )
        else:
            padding = 0

        chunk_flow_fw = _compute_flow(chunk_img0, chunk_img1)
        chunk_flow_bw = _compute_flow(chunk_img1, chunk_img0)
        chunk_flow_fw, chunk_flow_bw = jax.tree_map(
            lambda x: x[: x.shape[0] - padding],
            (chunk_flow_fw, chunk_flow_bw),
        )

        masked_flow_fw = np.where(chunk_mask0 == 1, chunk_flow_fw, np.inf)
        masked_flow_bw = np.where(chunk_mask1 == 1, chunk_flow_bw, np.inf)
        # Could be optimized further.
        chunk_occ_fw = common.parallel_map(
            compute_occ_brox, masked_flow_fw, masked_flow_bw
        ) + (1 - chunk_mask0)
        chunk_occ_bw = common.parallel_map(
            compute_occ_brox, masked_flow_bw, masked_flow_fw
        ) + (1 - chunk_mask1)
        results.append(
            {
                "flow_fw": chunk_flow_fw,
                "flow_bw": chunk_flow_bw,
                "occ_fw": chunk_occ_fw,
                "occ_bw": chunk_occ_bw,
            }
        )
    results = common.tree_collate(
        results, lambda *x: np.concatenate(x, axis=0)
    )
    return RAFTFlow(**results)
