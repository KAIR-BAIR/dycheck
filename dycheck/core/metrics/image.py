#!/usr/bin/env python3
#
# File   : image.py
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

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from dycheck.nn import functional as F


def compute_psnr(
    img0: jnp.ndarray, img1: jnp.ndarray, mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Compute PSNR between two images.

    Args:
        img0 (jnp.ndarray): An image of shape (H, W, 3) in float32.
        img1 (jnp.ndarray): An image of shape (H, W, 3) in float32.
        mask (Optional[jnp.ndarray]): An optional forground mask of shape (H,
            W, 1) in float32 {0, 1}. The metric is computed only on the pixels
            with mask == 1.

    Returns:
        jnp.ndarray: PSNR in dB of shape ().
    """
    mse = (img0 - img1) ** 2
    return -10.0 / jnp.log(10.0) * jnp.log(F.common.masked_mean(mse, mask))


def compute_ssim(
    img0: jnp.ndarray,
    img1: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> jnp.ndarray:
    """Computes SSIM between two images.

    This function was modeled after tf.image.ssim, and should produce
    comparable output.

    Image Inpainting for Irregular Holes Using Partial Convolutions.
        Liu et al., ECCV 2018.
        https://arxiv.org/abs/1804.07723

    Note that the mask operation is implemented as partial convolution. See
    Section 3.1.

    Args:
        img0 (jnp.ndarray): An image of size (H, W, 3) in float32.
        img1 (jnp.ndarray): An image of size (H, W, 3) in float32.
        mask (Optional[jnp.ndarray]): An optional forground mask of shape (H,
            W, 1) in float32 {0, 1}. The metric is computed only on the pixels
            with mask == 1.
        max_val (float): The dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
        filter_size (int): Size of the Gaussian blur kernel used to smooth the
            input images.
        filter_sigma (float): Standard deviation of the Gaussian blur kernel
            used to smooth the input images.
        k1 (float): One of the SSIM dampening parameters.
        k2 (float): One of the SSIM dampening parameters.

    Returns:
        jnp.ndarray: SSIM in range [0, 1] of shape ().
    """
    if mask is None:
        mask = jnp.ones_like(img0[..., :1])
    mask = mask[..., 0]  # type: ignore

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = jnp.exp(-0.5 * f_i)
    filt /= jnp.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, m, f):
        z_ = jsp.signal.convolve2d(
            z * m, f, mode="valid", precision=jax.lax.Precision.HIGHEST
        )
        m_ = jsp.signal.convolve2d(
            m,
            jnp.ones_like(f),
            mode="valid",
            precision=jax.lax.Precision.HIGHEST,
        )
        return jnp.where(m_ != 0, z_ * jnp.ones_like(f).sum() / m_, 0), (
            m_ != 0
        ).astype(z.dtype)

    filt_fn1 = lambda z, m: convolve2d(z, m, filt[:, None])
    filt_fn2 = lambda z, m: convolve2d(z, m, filt[None, :])

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn1 = jax.vmap(filt_fn1, in_axes=(2, None), out_axes=(2, None))
    filt_fn2 = jax.vmap(filt_fn2, in_axes=(2, None), out_axes=(2, None))
    filt_fn = lambda z, m: filt_fn1(*filt_fn2(z, m))

    mu0 = filt_fn(img0, mask)[0]
    mu1 = filt_fn(img1, mask)[0]
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2, mask)[0] - mu00
    sigma11 = filt_fn(img1**2, mask)[0] - mu11
    sigma01 = filt_fn(img0 * img1, mask)[0] - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = jnp.maximum(0.0, sigma00)
    sigma11 = jnp.maximum(0.0, sigma11)
    sigma01 = jnp.sign(sigma01) * jnp.minimum(
        jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = ssim_map.mean()

    return ssim


def get_compute_lpips() -> Callable[
    [np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray
]:
    """Get the LPIPS metric function.

    Note that torch and jax does not play well together. This means that
    running them in the same process on GPUs will cause issue.

    A workaround for now is to run torch on CPU only. For LPIPS computation,
    the overhead is not too bad.
    """

    import lpips
    import torch

    model = lpips.LPIPS(net="alex", spatial=True)

    @torch.inference_mode()
    def compute_lpips(
        img0: np.ndarray, img1: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.array:
        """Compute LPIPS between two images.

        This function computes mean LPIPS over masked regions. The input images
        are also masked. The following previous works leverage this metric:

        [1] Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic
        Scenes.
            Li et al., CVPR 2021.
            https://arxiv.org/abs/2011.13084

        [2] Transforming and Projecting Images into Class-conditional
        Generative Networks.
            Huh et al., CVPR 2020.
            https://arxiv.org/abs/2005.01703

        [3] Controlling Perceptual Factors in Neural Style Transfer.
            Gatys et al., CVPR 2017.
            https://arxiv.org/abs/1611.07865

        Args:
            img0 (jnp.ndarray): An image of shape (H, W, 3) in float32.
            img1 (jnp.ndarray): An image of shape (H, W, 3) in float32.
            mask (Optional[jnp.ndarray]): An optional forground mask of shape
                (H, W, 1) in float32 {0, 1}. The metric is computed only on the
                pixels with mask == 1.

        Returns:
            np.ndarray: LPIPS in range [0, 1] in shape ().
        """
        if mask is None:
            mask = jnp.ones_like(img0[..., :1])
        img0 = lpips.im2tensor(np.array(img0 * mask), factor=1 / 2)
        img1 = lpips.im2tensor(np.array(img1 * mask), factor=1 / 2)
        return F.common.masked_mean(
            model(img0, img1).cpu().numpy()[0, 0, ..., None], mask
        )

    return compute_lpips
