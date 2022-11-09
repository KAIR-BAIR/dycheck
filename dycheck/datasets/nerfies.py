#!/usr/bin/env python3
#
# File   : nerfies.py
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

import os.path as osp
from typing import Dict, Optional, Tuple

import cv2
import gin
import jax
import numpy as np
from absl import logging

from dycheck import geometry
from dycheck.nn import functional as F
from dycheck.utils import common, image, io, struct, types, visuals

from .base import Dataset, MetadataInfoMixin, Parser

SPLITS = [
    "train_intl",
    "train_mono",
    "val_intl",
    "val_mono",
    "train_common",
    "val_common",
]

DEFAULT_FACTORS: Dict[str, int] = {
    "nerfies/broom": 4,
    "nerfies/curls": 8,
    "nerfies/tail": 4,
    "nerfies/toby-sit": 4,
    "hypernerf/3dprinter": 4,
    "hypernerf/chicken": 4,
    "hypernerf/peel-banana": 4,
}
DEFAULT_FPS: Dict[str, float] = {
    "nerfies/broom": 15,
    "nerfies/curls": 5,
    "nerfies/tail": 15,
    "nerfies/toby-sit": 15,
    "hypernerf/3dprinter": 15,
    "hypernerf/chicken": 15,
    "hypernerf/peel-banana": 15,
}


def _load_scene_info(
    data_dir: types.PathType,
) -> Tuple[np.ndarray, float, float, float]:
    scene_dict = io.load(osp.join(data_dir, "scene.json"))
    center = np.array(scene_dict["center"], dtype=np.float32)
    scale = scene_dict["scale"]
    near = scene_dict["near"]
    far = scene_dict["far"]
    return center, scale, near, far


def _load_metadata_info(
    data_dir: types.PathType,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_dict = io.load(osp.join(data_dir, "dataset.json"))
    _frame_names = np.array(dataset_dict["ids"])

    metadata_dict = io.load(osp.join(data_dir, "metadata.json"))
    time_ids = np.array(
        [metadata_dict[k]["warp_id"] for k in _frame_names], dtype=np.uint32
    )
    camera_ids = np.array(
        [metadata_dict[k]["camera_id"] for k in _frame_names], dtype=np.uint32
    )

    frame_names_map = np.zeros(
        (time_ids.max() + 1, camera_ids.max() + 1), _frame_names.dtype
    )
    for i, (t, c) in enumerate(zip(time_ids, camera_ids)):
        frame_names_map[t, c] = _frame_names[i]

    return frame_names_map, time_ids, camera_ids


@gin.configurable()
class NerfiesParser(Parser):
    """Parser for the Nerfies dataset."""

    SPLITS = SPLITS

    def __init__(
        self,
        dataset: str,
        sequence: str,
        *,
        data_root: Optional[types.PathType] = None,
        factor: Optional[int] = None,
        fps: Optional[float] = None,
        use_undistort: bool = False,
    ):
        super().__init__(dataset, sequence, data_root=data_root)

        self._factor = factor or DEFAULT_FACTORS[f"{dataset}/{sequence}"]
        self._fps = fps or DEFAULT_FPS[f"{dataset}/{sequence}"]
        self.use_undistort = use_undistort

        (
            self._center,
            self._scale,
            self._near,
            self._far,
        ) = _load_scene_info(self.data_dir)
        (
            self._frame_names_map,
            self._time_ids,
            self._camera_ids,
        ) = _load_metadata_info(self.data_dir)
        self._load_extra_info()

        self.splits_dir = osp.join(self.data_dir, "splits")
        if not osp.exists(self.splits_dir):
            self._create_splits()

    def load_rgba(
        self,
        time_id: int,
        camera_id: int,
        *,
        use_undistort: Optional[bool] = None,
    ) -> np.ndarray:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        rgb_path = osp.join(
            self.data_dir,
            "rgb" if not use_undistort else "rgb_undistort",
            f"{self._factor}x",
            frame_name + ".png",
        )
        if osp.exists(rgb_path):
            rgba = io.load(rgb_path, flags=cv2.IMREAD_UNCHANGED)
            if rgba.shape[-1] == 3:
                rgba = np.concatenate(
                    [rgba, np.full_like(rgba[..., :1], 255)], axis=-1
                )
        elif use_undistort:
            camera = self.load_camera(time_id, camera_id, use_undistort=False)
            rgb = self.load_rgba(time_id, camera_id, use_undistort=False)[
                ..., :3
            ]
            rgb = cv2.undistort(rgb, camera.intrin, camera.distortion)
            alpha = (
                cv2.undistort(
                    np.full_like(rgb, 255),
                    camera.intrin,
                    camera.distortion,
                )
                == 255
            )[..., :1].astype(np.uint8) * 255
            rgba = np.concatenate([rgb, alpha], axis=-1)
            io.dump(rgb_path, rgba)
        else:
            raise ValueError(f"RGB image not found: {rgb_path}.")
        return rgba

    def load_camera(
        self,
        time_id: int,
        camera_id: int,
        *,
        use_undistort: Optional[bool] = None,
        **_,
    ) -> geometry.Camera:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        camera = (
            geometry.Camera.fromjson(
                osp.join(self.data_dir, "camera", frame_name + ".json")
            )
            .rescale_image_domain(1 / self._factor)
            .translate(-self._center)
            .rescale(self._scale)
        )
        if use_undistort:
            camera = camera.undistort_image_domain()
        return camera

    def load_covisible(
        self,
        time_id: int,
        camera_id: int,
        split: str,
        *,
        use_undistort: Optional[bool] = None,
        **_,
    ) -> np.ndarray:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        covisible_path = osp.join(
            self.data_dir,
            "covisible" if not use_undistort else "covisible_undistort",
            f"{self._factor}x",
            split,
            frame_name + ".png",
        )
        if osp.exists(covisible_path):
            # (H, W, 1) uint8 mask.
            covisible = io.load(covisible_path)[..., :1]
        elif use_undistort:
            camera = self.load_camera(time_id, camera_id, use_undistort=False)
            covisible = self.load_covisible(
                time_id,
                camera_id,
                split,
                use_undistort=False,
            ).repeat(3, axis=-1)
            alpha = (
                cv2.undistort(
                    np.full_like(covisible, 255),
                    camera.intrin,
                    camera.distortion,
                )
                == 255
            )[..., :1].astype(np.uint8) * 255
            covisible = cv2.undistort(
                covisible, camera.intrin, camera.distortion
            )[..., :1]
            covisible = ((covisible == 255) & (alpha == 255)).astype(
                np.uint8
            ) * 255
            io.dump(covisible_path, covisible)
        else:
            raise ValueError(
                f"Covisible image not found: {covisible_path}. If not "
                f"processed before, please consider running "
                f"tools/process_covisible.py."
            )
        return covisible

    def load_keypoints(
        self,
        time_id: int,
        camera_id: int,
        split: str,
        *,
        use_undistort: Optional[bool] = None,
        **_,
    ) -> np.ndarray:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        keypoints_path = osp.join(
            self.data_dir,
            "keypoint" if not use_undistort else "keypoint_undistort",
            f"{self._factor}x",
            split,
            frame_name + ".json",
        )
        if osp.exists(keypoints_path):
            camera = self.load_camera(
                time_id, camera_id, use_undistort=use_undistort
            )
            offset = 0.5 if camera.use_center else 0
            # (J, 3).
            keypoints = np.array(io.load(keypoints_path), np.float32)
        elif use_undistort:
            camera = self.load_camera(time_id, camera_id, use_undistort=False)
            offset = 0.5 if camera.use_center else 0
            keypoints = self.load_keypoints(
                time_id,
                camera_id,
                split,
                use_undistort=False,
            )
            keypoints = np.concatenate(
                [
                    camera.undistort_pixels(keypoints[:, :2]) - offset,
                    keypoints[:, -1:],
                ],
                axis=-1,
            )
            keypoints[keypoints[:, -1] == 0] = 0
            io.dump(keypoints_path, keypoints)
        else:
            raise ValueError(
                f"Keypoints not found: {keypoints_path}. If not "
                f"annotated before, please consider running "
                f"tools/annotate_keypoints.ipynb."
            )
        return np.concatenate(
            [keypoints[:, :2] + offset, keypoints[:, -1:]], axis=-1
        )

    def load_skeleton(
        self,
        split: str,
        *,
        use_undistort: Optional[bool] = None,
    ) -> visuals.Skeleton:
        if use_undistort is None:
            use_undistort = self.use_undistort

        skeleton_path = osp.join(
            self.data_dir,
            "keypoint",
            f"{self._factor}x",
            split,
            "skeleton.json",
        )
        if osp.exists(skeleton_path):
            skeleton = visuals.Skeleton(
                **{
                    k: v
                    for k, v in io.load(skeleton_path).items()
                    if k != "name"
                }
            )
        elif use_undistort:
            skeleton = self.load_skeleton(split, use_undistort=False)
            io.dump(skeleton_path, skeleton)
        else:
            raise ValueError(
                f"Skeleton not found: {skeleton_path}. If not "
                f"annotated before, please consider running "
                f"tools/annotate_keypoints.ipynb."
            )
        return skeleton

    def load_split(
        self, split: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert split in self.SPLITS

        split_dict = io.load(osp.join(self.splits_dir, f"{split}.json"))
        return (
            np.array(split_dict["frame_names"]),
            np.array(split_dict["time_ids"], np.uint32),
            np.array(split_dict["camera_ids"], np.uint32),
        )

    def load_bkgd_points(self) -> np.ndarray:
        bkgd_points = io.load(osp.join(self.data_dir, "points.npy")).astype(
            np.float32
        )
        bkgd_points = (bkgd_points - self._center) * self._scale
        return bkgd_points

    def _load_extra_info(self) -> None:
        extra_path = osp.join(self.data_dir, "extra.json")
        if osp.exists(extra_path):
            extra_dict = io.load(extra_path)
            bbox = np.array(extra_dict["bbox"], dtype=np.float32)
            lookat = np.array(extra_dict["lookat"], dtype=np.float32)
            up = np.array(extra_dict["up"], dtype=np.float32)
        else:
            cameras = common.parallel_map(
                self.load_camera, self._time_ids, self._camera_ids
            )
            bkgd_points = self.load_bkgd_points()
            points = np.concatenate(
                [
                    bkgd_points,
                    np.array([c.position for c in cameras], np.float32),
                ],
                axis=0,
            )
            bbox = np.stack([points.min(axis=0), points.max(axis=0)])
            lookat = geometry.utils.tringulate_rays(
                np.stack([c.position for c in cameras], axis=0),
                np.stack([c.optical_axis for c in cameras], axis=0),
            )
            up = np.mean([c.up_axis for c in cameras], axis=0)
            up /= np.linalg.norm(up)
            extra_dict = {
                "factor": self._factor,
                "fps": self._fps,
                "bbox": bbox.tolist(),
                "lookat": lookat.tolist(),
                "up": up.tolist(),
            }
            logging.info(
                f'Extra info not found. Dumping extra info to "{extra_path}."'
            )
            io.dump(extra_path, extra_dict)

        self._bbox = bbox
        self._lookat = lookat
        self._up = up

    def _create_splits(self):
        def _create_split(split):
            assert split in self.SPLITS, f'Unknown split "{split}".'

            # This should produce the same split in the original interleaving
            # data scheme.
            time_ids = self.uniq_time_ids
            camera_ids = self.uniq_camera_ids
            camera_ids_intl = np.tile(
                camera_ids, int(np.ceil(self.num_times / self.num_cameras))
            ).flatten()[: self.num_times]
            camera_ids_mono = np.zeros_like(time_ids)

            if split in ["train_intl", "val_intl"]:
                camera_ids = camera_ids_intl
            elif split in ["train_mono", "val_mono"]:
                camera_ids = camera_ids_mono
            elif split == "val_common":
                # Will be excluded.
                time_ids = np.tile(time_ids, 2)
                camera_ids = np.concatenate([camera_ids_intl, camera_ids_mono])
            elif split == "train_common":
                mask = camera_ids_intl == camera_ids_mono
                camera_ids = camera_ids_intl[mask]
                time_ids = time_ids[mask]

            if not split.startswith(
                "train"
            ):  # val_intl, val_mono, val_common.
                mask = np.zeros((self.num_times, self.num_cameras), bool)
                mask[time_ids] = True
                mask[time_ids, camera_ids] = False
                time_ids, camera_ids = np.mgrid[
                    : self.num_times, : self.num_cameras
                ]
                time_ids = time_ids[mask]
                camera_ids = camera_ids[mask]

            frame_names = self._frame_names_map[time_ids, camera_ids]
            split_dict = {
                "frame_names": frame_names,
                "time_ids": time_ids,
                "camera_ids": camera_ids,
            }
            io.dump(osp.join(self.splits_dir, f"{split}.json"), split_dict)

        _ = common.parallel_map(_create_split, self.SPLITS)

    @property
    def frame_names(self):
        return self._frame_names_map[self.time_ids, self.camera_ids]

    @property
    def time_ids(self):
        return self._time_ids

    @property
    def camera_ids(self):
        return self._camera_ids

    @property
    def center(self):
        return self._center

    @property
    def scale(self):
        return self._scale

    @property
    def near(self):
        return self._near

    @property
    def far(self):
        return self._far

    @property
    def factor(self):
        return self._factor

    @property
    def fps(self):
        return self._fps

    @property
    def bbox(self):
        return self._bbox

    @property
    def lookat(self):
        return self._lookat

    @property
    def up(self):
        return self._up


@gin.configurable()
class NerfiesDataset(Dataset, MetadataInfoMixin):
    """Nerfies dataset for both Nerfies and HyperNeRF sequences.

    Images might be undistorted during the loading process.

    The following previous works are tested on this dataset:

    [1] Nerfies: Deformable Neural Radiance Fields.
        Park et al., ICCV 2021.
        https://arxiv.org/abs/2011.12948

    [2] HyperNeRF: A Higher-Dimensional Representation for Topologically
    Varying Neural Radiance Fields.
        Park et al., SIGGRAPH Asia 2021.
        https://arxiv.org/abs/2106.13228
    """

    __parser_cls__: Parser = NerfiesParser

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Filtered by split.
        (
            self._frame_names,
            self._time_ids,
            self._camera_ids,
        ) = self.parser.load_split(self.split)

        # Preload cameras and points since it has small memory footprint.
        self.cameras = common.parallel_map(
            self.parser.load_camera, self._time_ids, self._camera_ids
        )

        if self.training:
            # If training, we need to make sure the unique metadata are
            # consecutive.
            self.validate_metadata_info()
        if self.bkgd_points_batch_size > 0:
            self.bkgd_points = self.parser.load_bkgd_points()
            # Weird batching logic from the original HyperNeRF repo.
            D = jax.local_device_count()
            self.bkgd_points_batch_size = min(
                self.bkgd_points_batch_size * D,
                len(self.bkgd_points),
            )
            self.bkgd_points_batch_size -= self.bkgd_points_batch_size % D

    def fetch_data(self, index: int):
        """Fetch the data (it maybe cached for multiple batches)."""
        time_id, camera_id = self.time_ids[index], self.camera_ids[index]

        camera = self.cameras[index]

        rgba = image.to_float32(self.parser.load_rgba(time_id, camera_id))
        rgb, mask = rgba[..., :3], rgba[..., -1:]

        rays = camera.pixels_to_rays(camera.get_pixels())._replace(
            metadata=struct.Metadata(
                time=np.full_like(rgb[..., :1], time_id, dtype=np.uint32),
                camera=np.full_like(rgb[..., :1], camera_id, dtype=np.uint32),
            )
        )

        data = {"rgb": rgb, "mask": mask, "rays": rays}
        if self.training:
            data = jax.tree_map(lambda x: x.reshape(-1, x.shape[-1]), data)
        else:
            try:
                # Covisible is not necessary for evaluation.
                data["covisible"] = image.to_float32(
                    self.parser.load_covisible(time_id, camera_id, self.split)
                )
            except ValueError:
                pass

        return data

    def preprocess(self, data: Dict):
        """Process the fetched / cached data with randomness.

        Note that in-place operations should be avoided since the raw data
        might be repeated for serveral times with different randomness.
        """
        if self.training:
            ray_inds = self.rng.choice(
                data["rays"].origins.shape[0],
                (self.batch_size,),
                replace=False,
            )
            batch = jax.tree_map(lambda x: x[ray_inds], data)
            if self.bkgd_points_batch_size > 0:
                point_inds = self.rng.choice(
                    self.bkgd_points.shape[0],
                    (self.bkgd_points_batch_size,),
                    replace=False,
                )
                bkgd_points = self.bkgd_points[point_inds]
                batch["bkgd_points"] = bkgd_points
        else:
            batch = data

        return batch

    @property
    def has_novel_view(self):
        return True

    @property
    def has_keypoints(self):
        return osp.exists(osp.join(self.data_dir, "keypoint"))

    @property
    def frame_names(self):
        return self._frame_names

    @property
    def time_ids(self):
        return self._time_ids

    @property
    def camera_ids(self):
        return self._camera_ids


@gin.configurable()
class NerfiesDatasetFromAllFrames(NerfiesDataset):
    """Nerfies dataset for both Nerfies and HyperNeRF sequences."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.training:
            rgbas = image.to_float32(
                np.array(
                    common.parallel_map(
                        self.parser.load_rgba, self._time_ids, self._camera_ids
                    )
                )
            ).reshape(-1, 4)
            self.rgbs, self.masks = rgbas[..., :3], rgbas[..., -1:]
            rays = jax.tree_map(
                lambda x: x.reshape(-1, x.shape[-1]),
                [
                    c.pixels_to_rays(c.get_pixels())._replace(
                        metadata=struct.Metadata(
                            time=np.full(
                                tuple(c.image_shape) + (1,),
                                ti,
                                dtype=np.uint32,
                            ),
                            camera=np.full(
                                tuple(c.image_shape) + (1,),
                                ci,
                                dtype=np.uint32,
                            ),
                        )
                    )
                    for c, ti, ci in zip(
                        self.cameras, self._time_ids, self._camera_ids
                    )
                ],
            )
            self.rays = struct.Rays(
                origins=np.concatenate([r.origins for r in rays], axis=0),
                directions=np.concatenate(
                    [r.directions for r in rays], axis=0
                ),
                metadata=struct.Metadata(
                    time=np.concatenate(
                        [r.metadata.time for r in rays], axis=0
                    ),
                    camera=np.concatenate(
                        [r.metadata.camera for r in rays], axis=0
                    ),
                ),
            )

    def fetch_data(self, index: int):
        """Fetch the data (it maybe cached for multiple batches)."""
        if not self.training:
            return super().fetch_data(index)

        return {"rgb": self.rgbs, "mask": self.masks, "rays": self.rays}

    def __next__(self):
        """Fetch cached data and preprocess."""
        batch = self._queue.get()
        return F.shard(batch) if self.training else F.to_device(batch)

    def run(self):
        """Main data fetching loop of the thread."""
        while True:
            if self.training:
                # Dummy index since we are going to load all.
                index = 0
            else:
                index = self._index
                self._index = self._index + 1
                self._index %= len(self)

            data = self.fetch_data(index)
            # Preprocess when loading rather than iterating.
            batch = self.preprocess(data)
            self._queue.put(batch)
