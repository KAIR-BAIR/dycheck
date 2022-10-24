#!/usr/bin/env python3
#
# File   : dense.py
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
from typing import Callable, Dict, Literal, Optional, Tuple

import gin
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dycheck.geometry import matv, se3
from dycheck.nn import MLP, EmbedPosEnc, PosEnc
from dycheck.nn import functional as F
from dycheck.utils import common, struct, types


@gin.configurable(denylist=["name"])
class TranslDensePosEnc(nn.Module):
    """A positional encoding layer that warps the input points before encoding
    through translation.
    """

    trunk_cls: Callable[..., nn.Module] = functools.partial(
        MLP,
        depth=6,
        width=128,
        output_init=jax.nn.initializers.uniform(scale=1e-4),
        output_channels=3,
        skips=(4,),
    )

    hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()

    # Metadata.
    num_embeddings: int = gin.REQUIRED
    points_embed_key: Literal["time"] = "time"
    points_embed_key_to: Literal["time_to"] = "time_to"
    points_embed_cls: Callable[..., nn.Module] = functools.partial(
        EmbedPosEnc,
        features=8,
        num_freqs=6,
        use_identity=True,
    )
    warped_points_embed_key: Optional[Literal["time"]] = None
    warped_points_embed_cls: Callable[..., nn.Module] = functools.partial(
        PosEnc,
        num_freqs=8,
        use_identity=True,
    )

    # Root-finding.
    max_iters: int = 50
    atol: float = 1e-5

    # Evaluation.
    # On-demand exclusion to save memory.
    exclude_fields: Tuple[str] = ()
    # On-demand returning to save memory. Will override exclusion.
    return_fields: Tuple[str] = ()

    def setup(self):
        assert self.points_embed_key is not None

        points_embed_cls = common.tolerant_partial(
            self.points_embed_cls, num_embeddings=self.num_embeddings
        )
        self.points_embed = points_embed_cls()
        warped_points_embed_cls = common.tolerant_partial(
            self.warped_points_embed_cls, num_embeddings=self.num_embeddings
        )
        self.warped_points_embed = warped_points_embed_cls()

        self.trunk = self.trunk_cls(hidden_init=self.hidden_init)

    def _eval(
        self,
        xs: jnp.ndarray,
        metadata: struct.Metadata,
        extra_params: Optional[struct.ExtraParams],
        use_warped_points_embed: bool = True,
        **_,
    ) -> Dict[str, jnp.ndarray]:
        assert self.points_embed_key in metadata._fields
        metadata = getattr(metadata, self.points_embed_key)
        points_embed = self.points_embed(
            xs=xs,
            metadata=metadata,
            alpha=getattr(extra_params, "warp_alpha")
            if extra_params
            else None,
        )
        warped_points = self.trunk(points_embed) + xs

        out = {"warped_points": warped_points}
        if use_warped_points_embed:
            out["warped_points_embed"] = self.warped_points_embed(
                warped_points
            )
        return out

    def warp_v2c(
        self,
        samples: struct.Samples,
        extra_params: Optional[struct.ExtraParams],
        return_jacobian: bool = False,
        use_warped_points_embed: bool = True,
        **_,
    ) -> Dict[str, jnp.ndarray]:
        assert samples.metadata is not None
        out = self._eval(
            samples.xs,
            samples.metadata,
            extra_params,
            use_warped_points_embed=use_warped_points_embed,
        )
        assert "warped_points" in out
        if return_jacobian:
            assert samples.xs.ndim == 2
            # TODO(Hang Gao @ 07/19): Need to test this.
            jac_fn = jax.vmap(
                jax.jacfwd(
                    lambda xs: self._eval(
                        xs,
                        samples.metadata,
                        extra_params,
                        use_warped_points_embed=False,
                    )["warped_points"],
                    in_axes=(0, 0, None),
                )
            )
            out["jacs"] = jac_fn(samples.xs)

        return out

    def warp_c2v(
        self,
        samples: struct.Samples,
        extra_params: Optional[struct.ExtraParams],
        init_points: jnp.ndarray,
        **_,
    ) -> Dict[str, jnp.ndarray]:
        assert (
            samples.metadata is not None
            and getattr(samples.metadata, self.points_embed_key_to) is not None
        )
        samples = samples._replace(
            metadata=struct.Metadata(
                **{
                    self.points_embed_key: getattr(
                        samples.metadata, self.points_embed_key_to
                    )
                }
            )
        )

        def _residual(warped_points: jnp.ndarray) -> jnp.ndarray:
            new_samples = samples._replace(xs=warped_points)
            cano_points = self.warp_v2c(
                new_samples, extra_params, use_warped_points_embed=False
            )["warped_points"]
            return samples.xs - cano_points

        solve_out = F.broyden.solve(
            _residual, init_points, self.max_iters, self.atol
        )
        return {
            "warped_points": solve_out["results"],
            "diffs": solve_out["diffs"][..., None],
            "converged": solve_out["converged"][..., None],
        }

    def __call__(
        self,
        samples: struct.Samples,
        extra_params: Optional[struct.ExtraParams],
        return_jacobian: bool = False,
        init_points: Optional[jnp.ndarray] = None,
        exclude_fields: Optional[Tuple[str]] = None,
        return_fields: Optional[Tuple[str]] = None,
        protect_fields: Tuple[str] = (),
        **_,
    ) -> Dict[str, jnp.ndarray]:
        """
        Args:
            samples (struct.Samples): The (...,) samples to be warped.
            extra_params (Optional[struct.ExtraParams]): The extra parameters.
            return_jacobian (bool): Whether to return the jacobian.
            init_points (Optional[jnp.ndarray]): The optional initial points to
                be used for root-finding.

        Returns:
            Dict[str, jnp.ndarray]: The warped points and auxilary information
                that might include jacobian.
        """
        if exclude_fields is None:
            exclude_fields = self.exclude_fields
        if return_fields is None:
            return_fields = self.return_fields

        assert samples.metadata is not None and (
            getattr(samples.metadata, self.points_embed_key) is not None
            or getattr(samples.metadata, self.points_embed_key_to) is not None
        )
        use_warp_v2c = (
            getattr(samples.metadata, self.points_embed_key) is not None
        )
        use_warp_c2v = (
            getattr(samples.metadata, self.points_embed_key_to) is not None
        )

        batch_shape = samples.xs.shape[:-1]
        samples = jax.tree_map(
            lambda x: x.reshape((np.prod(batch_shape), x.shape[-1])), samples
        )

        out, warp_out = {}, {}

        cano_points = samples.xs
        if use_warp_v2c:
            warp_out.update(
                **self.warp_v2c(
                    samples,
                    extra_params,
                    return_jacobian=return_jacobian,
                    use_warped_points_embed=not use_warp_c2v,
                )
            )
            cano_points = warp_out.pop("warped_points")

        warped_points = cano_points
        if use_warp_c2v:
            if init_points is None:
                # If root-finding initialization is not provided, use the
                # source points to start.
                init_points = samples.xs
            warped_samples = struct.Samples(
                xs=warped_points,
                directions=None,
                metadata=samples.metadata,
            )
            warp_out.update(
                **self.warp_c2v(
                    warped_samples,
                    extra_params,
                    init_points=init_points,
                )
            )
            warped_points = warp_out.pop("warped_points")

        out = {"cano_points": cano_points, "warped_points": warped_points}
        # Only save the last warp auxiliary output.
        out.update(**warp_out)

        out = jax.tree_map(lambda x: x.reshape(batch_shape + x.shape[1:]), out)

        out = common.traverse_filter(
            out,
            exclude_fields=exclude_fields,
            return_fields=return_fields,
            protect_fields=protect_fields,
            inplace=True,
        )

        return out


@gin.configurable(denylist=["name"])
class SE3DensePosEnc(TranslDensePosEnc):
    """A positional encoding layer that warps the input points before encoding
    through rotation and translation.
    """

    trunk_cls: Callable[..., nn.Module] = functools.partial(
        MLP,
        depth=6,
        width=128,
        skips=(4,),
    )

    rotation_depth: int = 0
    rotation_width: int = 128
    rotation_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)

    transl_depth: int = 0
    transl_width: int = 128
    transl_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)

    hidden_init: types.Initializer = jax.nn.initializers.xavier_uniform()

    def setup(self):
        super().setup()

        self.branches = {
            "rotation": MLP(
                depth=self.rotation_depth,
                width=self.rotation_width,
                hidden_init=self.hidden_init,
                output_init=self.rotation_init,
                output_channels=3,
            ),
            "transl": MLP(
                depth=self.transl_depth,
                width=self.transl_width,
                hidden_init=self.hidden_init,
                output_init=self.transl_init,
                output_channels=3,
            ),
        }

    def _eval(
        self,
        xs: jnp.ndarray,
        metadata: struct.Metadata,
        extra_params: Optional[struct.ExtraParams],
        use_warped_points_embed: bool = True,
        **_,
    ) -> Dict[str, jnp.ndarray]:
        assert self.points_embed_key in metadata._fields
        metadata = getattr(metadata, self.points_embed_key)
        points_embed = self.points_embed(
            xs=xs,
            metadata=metadata,
            alpha=getattr(extra_params, "warp_alpha")
            if extra_params
            else None,
        )
        trunk = self.trunk(points_embed)

        rotation = self.branches["rotation"](trunk)
        transl = self.branches["transl"](trunk)
        theta = jnp.linalg.norm(rotation, axis=-1)
        rotation = rotation / theta[..., None]
        transl = transl / theta[..., None]
        screw_axis = jnp.concatenate([rotation, transl], axis=-1)
        transform = se3.exp_se3(screw_axis, theta)

        warped_points = se3.from_homogenous(
            matv(transform, se3.to_homogenous(xs))
        )

        out = {"warped_points": warped_points}
        if use_warped_points_embed:
            out["warped_points_embed"] = self.warped_points_embed(
                warped_points
            )
        return out
