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
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F

from dycheck.utils import common, image

from . import _impl


def get_mttr() -> Tuple[torch.nn.Module, Callable]:
    from transformers import logging

    logging.set_verbosity_warning()
    return torch.hub.load("mttr2021/MTTR:main", "mttr_refer_youtube_vos")


def get_compute_mttr_video_mask(
    show_pbar: bool = True,
    desc: Optional[str] = "* Computing MTTR video mask",
    **kwargs,
) -> Callable[
    [np.ndarray, Sequence[str], Optional[np.ndarray]], List[np.ndarray]
]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, postprocessor = get_mttr()
    return functools.partial(
        compute_mttr_video_mask,
        model.to("cuda").eval(),
        postprocessor,
        show_pbar=show_pbar,
        desc=desc,
        **kwargs,
    )


@torch.inference_mode()
def compute_mttr_video_mask(
    model: torch.nn.Module,
    postprocessor: Callable,
    video: np.ndarray,
    prompts: Sequence[str],
    masks: Optional[np.ndarray] = None,
    *,
    window_length: int = 24,
    window_overlap: int = 6,
    show_pbar: bool = True,
    desc: Optional[str] = "* Compute MTTR video mask",
) -> List[np.ndarray]:
    """Segment video mask with MTTR model given a set of prompts.

    End-to-End Referring Video Object Segmentation with Multimodal
    Transformers.
        Botach et al., CVPR 2022.
        https://arxiv.org/abs/2111.14821

    Note that RAFT model takes uint8 (or 255-scale float32) images as
    input.

    Args:
        video (np.ndarray): (T, H, W, 3), a video float32 or uint8 RGB format.
        prompts (Sequence[str]): a set of prompts for the video segmentation.
        masks (Optional[np.ndarray]): (T, H, W, 1), a video of binary mask
            indicating the valid regions.
        window_length (int): the length of the sliding window.
        window_overlap (int): the overlap of the sliding window.

    Returns:
        pred_masks_per_query (List[np.ndarray]): a list of (T, H, W, 1) float32
            segmented masks indicating foreground (where the objects referred
            by prompts are).
    """
    assert len(prompts) in [1, 2], "MTTR expects 1 or 2 prompts."

    # Ignore warnings in the forward pass of MTTR.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        video = (
            torch.from_numpy(image.to_float32(video))
            .permute(0, 3, 1, 2)
            .float()
            .to("cuda")
        )
        input_video = F.resize(video, size=360, max_size=640)
        input_video = F.normalize(
            input_video, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        video_metadata = {
            "resized_frame_size": input_video.shape[-2:],
            "original_frame_size": video.shape[-2:],
        }

        # partition the clip into overlapping windows of frames:
        windows = [
            input_video[i : i + window_length]
            for i in range(0, len(input_video), window_length - window_overlap)
        ]
        # clean up the text queries:
        prompts = [" ".join(q.lower().split()) for q in prompts]

        pred_masks_per_query = []
        T, _, H, W = video.shape
        for prompt in (common.tqdm if show_pbar else lambda x, **_,: x)(
            prompts, desc=desc, position=0
        ):
            pred_masks = torch.zeros(size=(T, 1, H, W))
            for i, window in enumerate(
                (common.tqdm if show_pbar else lambda x, **_,: x)(
                    windows, position=1, leave=False
                )
            ):
                window = _impl.nested_tensor_from_videos_list([window])
                valid_indices = torch.arange(len(window.tensors)).to("cuda")
                outputs = model(window, valid_indices, [prompt])
                window_masks = postprocessor(
                    outputs, [video_metadata], window.tensors.shape[-2:]
                )[0]["pred_masks"]
                win_start_idx = i * (window_length - window_overlap)
                pred_masks[
                    win_start_idx : win_start_idx + window_length
                ] = window_masks
            pred_masks = pred_masks.cpu().numpy().transpose(0, 2, 3, 1)
            if masks is not None:
                pred_masks = pred_masks * masks
            pred_masks_per_query.append(pred_masks)

    return pred_masks_per_query
