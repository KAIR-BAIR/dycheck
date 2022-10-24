#!/usr/bin/env python3
#
# File   : nerf.py
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
from typing import Callable, Dict, Literal, Mapping, Optional, Sequence, Tuple

import gin
import jax.numpy as jnp
from flax import linen as nn
from jax import random

from dycheck import geometry
from dycheck.nn import Embed, NeRFMLP, PosEnc
from dycheck.nn import functional as F
from dycheck.utils import common, struct, types


@gin.configurable(denylist=["name"])
class DyNeRF(nn.Module):
    # Data specifics.
    embeddings_dict: Mapping[
        Literal["time", "camera"], Sequence[int]
    ] = gin.REQUIRED
    near: float = gin.REQUIRED
    far: float = gin.REQUIRED

    # Architecture.
    use_warp: bool = False
    points_embed_key: Literal["time"] = "time"
    points_embed_cls: Callable[..., nn.Module] = functools.partial(
        Embed,
        features=8,
    )
    rgb_embed_key: Optional[Literal["time", "camera"]] = None
    rgb_embed_cls: Callable[..., nn.Module] = functools.partial(
        Embed,
        features=8,
    )
    use_viewdirs: bool = False
    viewdirs_embed_cls: Callable[..., nn.Module] = functools.partial(
        PosEnc,
        num_freqs=4,
        use_identity=True,
    )
    sigma_activation: types.Activation = nn.softplus

    # Rendering.
    num_coarse_samples: int = 128
    num_fine_samples: int = 128
    use_randomized: bool = True
    noise_std: Optional[float] = None
    use_white_bkgd: bool = False
    use_linear_disparity: bool = False
    use_sample_at_infinity: bool = True
    use_cull_cameras: bool = False
    cameras_dict: Optional[
        Dict[
            Literal[
                "intrin",
                "extrin",
                "radial_distortion",
                "tangential_distortion",
                "image_size",
            ],
            jnp.ndarray,
        ]
    ] = None
    num_min_frames: int = 5
    min_frame_ratio: float = 0.1

    # Biases.
    #  resample_padding: float = 0.01
    #  sigma_bias: float = -1
    #  rgb_padding: float = 0.001
    resample_padding: float = 0.0
    sigma_bias: float = 0.0
    rgb_padding: float = 0.0

    # Evaluation.
    # On-demand exclusion to save memory.
    exclude_fields: Tuple[str] = ()
    # On-demand returning to save memory. Will override exclusion.
    return_fields: Tuple[str] = ()

    @property
    def use_fine(self) -> bool:
        return self.num_fine_samples > 0

    @property
    def num_points_embeds(self):
        return max(self.embeddings_dict[self.points_embed_key]) + 1

    @property
    def use_rgb_embed(self) -> bool:
        return self.rgb_embed_key is not None

    @property
    def num_rgb_embeds(self):
        return max(self.embeddings_dict[self.rgb_embed_key]) + 1

    def setup(self):
        points_embed_cls = common.tolerant_partial(
            self.points_embed_cls, num_embeddings=self.num_points_embeds
        )
        self.points_embed = points_embed_cls()

        if self.use_rgb_embed:
            rgb_embed_cls = common.tolerant_partial(
                self.rgb_embed_cls, num_embeddings=self.num_rgb_embeds
            )
            self.rgb_embed = rgb_embed_cls()

        if self.use_viewdirs:
            self.viewdirs_embed = self.viewdirs_embed_cls()

        nerfs = {"coarse": NeRFMLP()}
        if self.use_fine:
            nerfs["fine"] = NeRFMLP()
        self.nerfs = nerfs

    def get_conditions(
        self, samples: struct.Samples
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        trunk_conditions, rgb_conditions = [], []

        if self.rgb_embed_key is not None:
            assert samples.metadata is not None
            rgb_embed = getattr(samples.metadata, self.rgb_embed_key)
            rgb_embed = self.rgb_embed(rgb_embed)
            rgb_conditions.append(rgb_embed)

        if self.use_viewdirs:
            viewdirs_embed = self.viewdirs_embed(samples.directions)
            rgb_conditions.append(viewdirs_embed)

        trunk_conditions = (
            jnp.concatenate(trunk_conditions, axis=-1)
            if trunk_conditions
            else None
        )
        rgb_conditions = (
            jnp.concatenate(rgb_conditions, axis=-1)
            if rgb_conditions
            else None
        )
        return trunk_conditions, rgb_conditions

    def embed_samples(
        self,
        samples: struct.Samples,
        extra_params: Optional[struct.ExtraParams],
        use_warp: Optional[bool] = None,
        use_warp_jacobian: bool = False,
    ):
        if use_warp is None:
            use_warp = self.use_warp
        else:
            assert self.use_warp, "The model does not support warping."

        if use_warp:
            warp_out = self.points_embed(
                samples, extra_params, return_jacobian=use_warp_jacobian
            )
            assert "warped_points_embed" in warp_out
            warped_points_embed = warp_out.pop("warped_points_embed")
        else:
            # Return original points if no warp has been applied.
            warp_out = {"warped_points": samples.xs}
            if self.use_warp:
                warped_points_embed = self.points_embed.warped_points_embed(
                    xs=samples.xs
                )
            else:
                warped_points_embed = self.points_embed(
                    xs=samples.xs,
                    metadata=getattr(samples.metadata, self.points_embed_key),
                )

        return warped_points_embed, warp_out

    def eval_samples(
        self,
        samples: struct.Samples,
        extra_params: Optional[struct.ExtraParams],
        use_warp: Optional[bool] = None,
        use_warp_jacobian: bool = False,
        level: Optional[Literal["coarse", "fine"]] = None,
        use_randomized: Optional[bool] = None,
        exclude_fields: Optional[Tuple[str]] = None,
        return_fields: Optional[Tuple[str]] = None,
        protect_fields: Tuple[str] = (),
    ) -> Dict[str, jnp.ndarray]:
        """Evaluate points at given positions.

        Assumes (N, S, 3) points, (N, S, 3) viewdirs, (N, S, 1) metadata.

        Supported fields:
             - point_sigma
             - point_rgb
             - point_feat
             - points
             - warped_points
             - warp_out
        """
        if use_randomized is None:
            use_randomized = self.use_randomized
        if exclude_fields is None:
            exclude_fields = self.exclude_fields
        if return_fields is None:
            return_fields = self.return_fields

        nerf = self.nerfs[
            level
            if level is not None
            else ("fine" if self.use_fine else "coarse")
        ]
        trunk_conditions, rgb_conditions = self.get_conditions(samples)

        out = {"points": samples.xs}

        warped_points_embed, warp_out = self.embed_samples(
            samples,
            extra_params,
            use_warp=use_warp,
            use_warp_jacobian=use_warp_jacobian,
        )
        out["warp_out"] = warp_out

        logits = nerf(warped_points_embed, trunk_conditions, rgb_conditions)
        # Perturb logits with noise. Disabled by default (noise_std is None).
        logits = F.rendering.perturb_logits(
            self.make_rng(level), logits, use_randomized, self.noise_std
        )

        # Apply activations.
        logits["point_sigma"] = self.sigma_activation(
            logits["point_sigma"] + self.sigma_bias
        )
        if "point_rgb" in logits:
            logits["point_rgb"] = (
                nn.sigmoid(logits["point_rgb"]) * (1 + 2 * self.rgb_padding)
                - self.rgb_padding
            )
        out.update(logits)

        out = common.traverse_filter(
            out,
            exclude_fields=exclude_fields,
            return_fields=return_fields,
            protect_fields=protect_fields,
            inplace=True,
        )
        return out

    def render_samples(
        self,
        samples: struct.Samples,
        extra_params: Optional[struct.ExtraParams],
        use_warp: Optional[bool] = None,
        use_warp_jacobian: bool = False,
        level: Optional[str] = None,
        use_randomized: Optional[bool] = None,
        bkgd_rgb: Optional[jnp.ndarray] = None,
        use_cull_cameras: Optional[bool] = None,
        exclude_fields: Optional[str] = None,
        return_fields: Optional[str] = None,
        protect_fields: Tuple[str] = (),
    ) -> Dict[str, jnp.ndarray]:
        """
        Note that tvals is of shape (N, S + 1) such that it covers the start
        and end of the ray. Samples are evaluated at mid points.

        Supported fields:
            - point_sigma
            - point_rgb
            - point_feat
            - points
            - warped_points
            - warp_out
            - rgb
            - depth
            - med_depth
            - acc
            - alpha
            - trans
            - weights
        """
        if use_cull_cameras is None:
            use_cull_cameras = self.use_cull_cameras
        if exclude_fields is None:
            exclude_fields = self.exclude_fields
        if return_fields is None:
            return_fields = self.return_fields

        # Return all fields and filter after.
        out = self.eval_samples(
            samples,
            extra_params,
            use_warp=use_warp,
            use_warp_jacobian=use_warp_jacobian,
            level=level,
            use_randomized=use_randomized,
        )

        if use_cull_cameras and self.cameras_dict is not None:
            # Cull out the samples that are outside of the camera frustrums.
            # This is particularly useful to avoid rendering points that are
            # not supervised during training.
            pixels = geometry.project(
                out["points"][None],
                self.cameras_dict["intrin"][:, None, None],
                self.cameras_dict["extrin"][:, None, None],
                self.cameras_dict["radial_distortion"][:, None, None],
                self.cameras_dict["tangential_distortion"][:, None, None],
            )
            mask = jnp.all(
                (pixels < self.cameras_dict["image_size"][:, None, None])
                & (pixels >= 0),
                axis=-1,
                keepdims=True,
            ).astype(jnp.float32)
            mask = mask.sum(axis=0) > max(
                self.num_min_frames, self.min_frame_ratio * len(mask)
            )
            out["point_sigma"] = jnp.where(mask, out["point_sigma"], 0)

        if bkgd_rgb is None:
            bkgd_rgb = jnp.full(
                (3,), 1 if self.use_white_bkgd else 0, dtype=jnp.float32
            )
        out.update(
            F.rendering.volrend(
                out,
                samples,
                bkgd_rgb=bkgd_rgb,
                use_sample_at_infinity=self.use_sample_at_infinity,
            )
        )

        out = common.traverse_filter(
            out,
            exclude_fields=exclude_fields,
            return_fields=return_fields,
            protect_fields=protect_fields,
            inplace=True,
        )

        return out

    def __call__(
        self,
        rays: struct.Rays,
        extra_params: Optional[struct.ExtraParams],
        use_warp: Optional[bool] = None,
        use_warp_jacobian: bool = False,
        use_randomized: Optional[bool] = None,
        bkgd_rgb: Optional[jnp.ndarray] = None,
        use_cull_cameras: Optional[bool] = None,
        exclude_fields: Optional[Tuple[str]] = None,
        return_fields: Optional[Tuple[str]] = None,
        protect_fields: Tuple[str] = (),
    ) -> Dict[str, Dict[str, jnp.ndarray]]:
        """
        Supported fields:
            - coarse/fine
                - point_sigma
                - point_rgb
                - point_feat
                - points
                - warped_points
                - warp_out
                    - scores
                    - weights
                - rgb
                - depth
                - med_depth
                - acc
                - alpha
                - trans
                - weights
                - tvals
        """
        if use_randomized is None:
            use_randomized = self.use_randomized
        if exclude_fields is None:
            exclude_fields = self.exclude_fields
        if return_fields is None:
            return_fields = self.return_fields

        # Brute-force adding prefix to return_fields if there's any. It will be
        # used by the training loop.
        return_fields = (
            return_fields
            + tuple([f"coarse/{f}" for f in return_fields])  # type: ignore
            + tuple([f"fine/{f}" for f in return_fields])  # type: ignore
        )

        render_samples = functools.partial(
            self.render_samples,
            extra_params=extra_params,
            use_warp=use_warp,
            use_randomized=use_randomized,
            bkgd_rgb=bkgd_rgb,
            use_cull_cameras=use_cull_cameras,
        )

        # Sample coarse points and render rays.
        samples = F.sampling.uniform(
            self.make_rng("coarse"),
            rays,
            self.num_coarse_samples,
            self.near,
            self.far,
            use_randomized=use_randomized,
            use_linear_disparity=self.use_linear_disparity,
        )
        coarse_out = render_samples(
            samples=samples,
            level="coarse",
            protect_fields=("weights",) if self.use_fine else (),
        )
        out = {"coarse": coarse_out}
        # (N, S, 1). Note that this is different from the mipNeRF
        # implementation because of "sampling at infinity". In this case, the
        # sample at the last position of each ray (at infinity) is effectively
        # dropped.
        out["coarse"]["tvals"] = samples.tvals

        # Sample fine points and render rays.
        if self.use_fine:
            assert samples.tvals is not None
            samples = F.sampling.ipdf(
                self.make_rng("fine"),
                0.5 * (samples.tvals[..., 1:, 0] + samples.tvals[..., :-1, 0]),
                coarse_out["weights"][..., 1:-1, 0],
                rays,
                samples,
                self.num_fine_samples,
                use_randomized=use_randomized,
                #  self.resample_padding,
            )
            out["fine"] = render_samples(
                samples=samples,
                use_warp_jacobian=use_warp_jacobian,
                level="fine",
            )
            out["fine"]["tvals"] = samples.tvals

        out = common.traverse_filter(
            out,
            exclude_fields=exclude_fields,
            return_fields=return_fields,
            protect_fields=protect_fields,
            inplace=True,
        )

        return out

    @classmethod
    def create(
        cls,
        key: types.PRNGKey,
        embeddings_dict: Dict[Literal["time", "camera"], Sequence[int]],
        near: float,
        far: float,
        cameras_dict: Optional[
            Dict[
                Literal[
                    "intrin",
                    "extrin",
                    "radial_distortion",
                    "tangential_distortion",
                    "image_size",
                ],
                jnp.ndarray,
            ]
        ] = None,
        exclude_fields: Tuple[str] = (),
        return_fields: Tuple[str] = (),
    ):
        """Neural Randiance Field.

        Args:
            key (PRNGKey): PRNG key.
            embeddings_dict (Dict[str, Sequence[int]]): Dictionary of unique
                embeddings.
            near (float): Near plane.
            far (float): Far plane.
            exclude_fields (Tuple[str]): Fields to exclude.
            return_fields (Tuple[str]): Fields to return.

        Returns:
            model (nn.Model): the dynamic NeRF model.
            params (Dict[str, jnp.ndarray]): the parameters for the model.
        """

        model = cls(
            embeddings_dict,
            near=near,
            far=far,
            cameras_dict=cameras_dict,
            exclude_fields=exclude_fields,
            return_fields=return_fields,
        )

        rays = struct.Rays(
            origins=jnp.ones((1, 3), jnp.float32),
            directions=jnp.ones((1, 3), jnp.float32),
            metadata=struct.Metadata(
                time=jnp.ones((1, 1), jnp.uint32),
                camera=jnp.ones((1, 1), jnp.uint32),
            ),
        )
        extra_params = struct.ExtraParams(
            warp_alpha=jnp.zeros((1,), jnp.float32),
            ambient_alpha=jnp.zeros((1,), jnp.float32),
        )

        key, key0, key1 = random.split(key, 3)
        variables = model.init(
            {"params": key, "coarse": key0, "fine": key1},
            rays=rays,
            extra_params=extra_params,
        )

        return model, variables
