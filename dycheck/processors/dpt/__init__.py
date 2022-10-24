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
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

from dycheck.utils import image, path_ops

from ._impl.models import DPTDepthModel
from ._impl.transforms import NormalizeImage, PrepareForNet, Resize


def get_dpt() -> Tuple[torch.nn.Module, Callable]:
    model_dir = osp.join(torch.hub.get_dir(), "checkpoints/dpt")
    model_path = osp.join(model_dir, "dpt_large-midas-2f21e586.pt")
    if not osp.exists(model_dir):
        path_ops.mkdir(model_dir)
        torch.hub.download_url_to_file(
            "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
            model_path,
        )
    model = DPTDepthModel(
        path=model_path,
        backbone="vitl16_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    netW = netH = 384
    transform = Compose(
        [
            Resize(
                netW,
                netH,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )
    return model, transform


def get_compute_dpt_disp() -> Callable[
    [np.ndarray, Optional[np.ndarray]], np.ndarray
]:
    model, transform = get_dpt()
    return functools.partial(
        compute_dpt_disp,
        model.eval().to(memory_format=torch.channels_last).half().to("cuda"),
        transform,
    )


@torch.inference_mode()
def compute_dpt_disp(
    model: DPTDepthModel,
    transform: Callable,
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate monocular disparity with DPT model for a single image.

    Vision Transformers for Dense Prediction.
        Ranftl et al., ICCV 2021.
        https://arxiv.org/abs/2103.13413

    Args:
        img (np.ndarray): (H, W, 3), a image in float32 or uint8 RGB format.
        mask (Optional[np.ndarray]): (H, W, 1), a binary mask of the image.

    Returns:
        pred_disp (np.ndarray): (H, W, 1), a monocular disparity map.
    """

    x = torch.from_numpy(transform({"image": image.to_float32(img)})["image"])[
        None
    ].to("cuda")
    x = x.to(memory_format=torch.channels_last).half()

    pred_disp = model.forward(x)
    pred_disp = (
        torch.nn.functional.interpolate(
            pred_disp.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )[..., None]
    if mask is not None:
        pred_disp = pred_disp * mask
    return pred_disp
