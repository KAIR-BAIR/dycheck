#!/usr/bin/env python3
#
# File   : _impl.py
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

from typing import NamedTuple

import numpy as np
import torch


class NestedTensor(NamedTuple):
    tensors: torch.Tensor
    mask: torch.Tensor


def nested_tensor_from_videos_list(videos_list):
    def _max_by_axis(the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    max_size = _max_by_axis([list(img.shape) for img in videos_list])
    padded_batch_shape = [len(videos_list)] + max_size
    B, T, _, H, W = padded_batch_shape
    dtype = videos_list[0].dtype
    device = videos_list[0].device
    padded_videos = torch.zeros(padded_batch_shape, dtype=dtype, device=device)
    videos_pad_masks = torch.ones(
        (B, T, H, W), dtype=torch.bool, device=device
    )
    for vid_frames, pad_vid_frames, vid_pad_m in zip(
        videos_list, padded_videos, videos_pad_masks
    ):
        pad_vid_frames[
            : vid_frames.shape[0],
            :,
            : vid_frames.shape[2],
            : vid_frames.shape[3],
        ].copy_(vid_frames)
        vid_pad_m[
            : vid_frames.shape[0], : vid_frames.shape[2], : vid_frames.shape[3]
        ] = False
    return NestedTensor(
        padded_videos.transpose(0, 1), videos_pad_masks.transpose(0, 1)
    )


def apply_mask(image, mask, color, transparency=0.7):
    mask = mask[..., None].repeat(repeats=3, axis=2)
    mask = mask * transparency
    color_matrix = np.ones(image.shape, dtype=np.float) * color
    out_image = color_matrix * mask + image * (1.0 - mask)
    return out_image
