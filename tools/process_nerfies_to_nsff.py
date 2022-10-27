#!/usr/bin/env python3
#
# File   : process_nerfies_to_nsff.py
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
import os.path as osp
from typing import Callable, Optional, Sequence

import gin
import numpy as np
from absl import app, flags, logging

from dycheck import core
from dycheck.datasets import Parser
from dycheck.processors import colmap
from dycheck.utils import common, io, path_ops, types

flags.DEFINE_multi_string(
    "gin_configs", None, "Gin config files.", required=True
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
FLAGS = flags.FLAGS


@gin.configurable(module="process_nerfies_to_nsff")
@dataclasses.dataclass
class Config(object):
    parser_cls: Callable[..., Parser] = gin.REQUIRED
    train_split: str = gin.REQUIRED
    val_split: str = gin.REQUIRED
    val_common_split: Optional[str] = None
    keypoint_splits: Sequence[str] = ()
    dump_suffix: str = ""
    dump_root: types.PathType = osp.join(
        osp.dirname(__file__), "../../nsff_pl/data"
    )


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

    logging.info("*** Starting processing Nerfies to NSFF data format.")
    parser = config.parser_cls()
    assert not getattr(
        parser, "use_undistort", True
    ), "NSFF expects undistorted images."

    dump_dir = osp.join(
        config.dump_root, parser.dataset, parser.sequence + config.dump_suffix
    )

    train_frame_names = parser.load_split(config.train_split)[0]
    for split in config.keypoint_splits:
        dir_suffix = {
            "train": "",
            "train_mono": "",
            "train_common": "_common",
        }[split]
        keypoint_dir = osp.join(
            parser.data_dir, "keypoint", f"{parser._factor}x", split
        )
        keypoint_dump_dir = osp.join(dump_dir, "keypoints" + dir_suffix)
        keypoint_paths = [
            p
            for p in path_ops.ls(osp.join(keypoint_dir, "*.json"))
            if osp.basename(p) != "skeleton.json"
        ]
        path_ops.mkdir(keypoint_dump_dir)
        for keypoint_path in keypoint_paths:
            name = path_ops.basename(keypoint_path, with_ext=False)
            i = np.arange(len(train_frame_names))[train_frame_names == name][0]
            path_ops.cp(
                keypoint_path, osp.join(keypoint_dump_dir, f"{i:05d}.json")
            )
        path_ops.cp(
            osp.join(keypoint_dir, "skeleton.json"),
            osp.join(keypoint_dump_dir, "skeleton.json"),
        )

    splits = [config.train_split, config.val_split]
    trainings = [True, False]
    dir_suffixes = ["", "_test"]
    if config.val_common_split is not None:
        splits.append(config.val_common_split)
        trainings.append(False)
        dir_suffixes.append("_test_common")

    for split, training, dir_suffix in zip(splits, trainings, dir_suffixes):
        colmap_dir = osp.join(dump_dir, "sparse" + dir_suffix, "0")
        img_dir = osp.join(dump_dir, "images" + dir_suffix)
        mask_dir = osp.join(dump_dir, "masks" + dir_suffix)
        metadata_path = osp.join(dump_dir, f"metadata{dir_suffix}.json")

        frame_names, time_ids, camera_ids = parser.load_split(split)

        # COLMAP SfM data to fill in.
        cameras, images, points3D = {}, {}, {}
        metadata_dict = {}
        for i, (_, time_id, camera_id) in enumerate(
            zip(
                common.tqdm(frame_names, desc=f"* Processing {split}"),
                time_ids,
                camera_ids,
            )
        ):
            new_frame_name = f"{i:05d}.png"

            # Dump images and masks.
            rgba = parser.load_rgba(time_id, camera_id)
            rgb = rgba[..., :3]
            mask = (
                rgba[..., 3:]
                if training
                else parser.load_covisible(time_id, camera_id, split)
            )
            io.dump(osp.join(img_dir, new_frame_name), rgb)
            io.dump(osp.join(mask_dir, new_frame_name), mask)

            camera = parser.load_camera(time_id, camera_id)

            # Intrinsics.
            cameras[new_frame_name] = colmap.Camera(
                id=i,
                model="OPENCV",
                width=int(camera.image_size_x),
                height=int(camera.image_size_y),
                params=np.array(
                    [
                        float(camera.scale_factor_x),
                        float(camera.scale_factor_y),
                        float(camera.principal_point_x),
                        float(camera.principal_point_y),
                        float(camera.radial_distortion[0]),
                        float(camera.radial_distortion[1]),
                        float(camera.tangential_distortion[0]),
                        float(camera.tangential_distortion[1]),
                        # Nerfies & HyperNeRF will have zero k3 anyway.
                        #  float(camera.radial_distortion[2]),
                    ]
                ),
            )

            # Points. Only placeholder and won't be used at all.
            point3D_ids = [0]
            for pi in point3D_ids:
                points3D[pi] = colmap.Point3D(
                    id=pi,
                    xyz=np.zeros((3,), dtype=np.float32),
                    rgb=np.zeros((3,), dtype=np.uint8),
                    error=0.0,
                    image_ids=np.array([i]),
                    point2D_idxs=np.array([0]),
                )

            # Extrinsics.
            images[new_frame_name] = colmap.Image(
                id=i,
                qvec=colmap.rotmat2qvec(camera.orientation),
                tvec=camera.translation,
                camera_id=i,
                name=new_frame_name,
                xys=np.zeros((len(point3D_ids), 2)),  # Not used.
                point3D_ids=point3D_ids,  # Not used.
            )

            # Metadata.
            metadata_dict[new_frame_name.replace(".png", "")] = {
                "time": int(time_id),
                "camera": int(camera_id),
            }

        # Dump COLMAP SfM data.
        path_ops.mkdir(colmap_dir)
        colmap.write_cameras_binary(
            cameras, osp.join(colmap_dir, "cameras.bin")
        )
        colmap.write_images_binary(images, osp.join(colmap_dir, "images.bin"))
        colmap.write_points3D_binary(
            points3D, osp.join(colmap_dir, "points3D.bin")
        )

        # Dump metadata information.
        io.dump(metadata_path, metadata_dict)

    # Dump extra information.
    extra_dict = {
        "fps": parser.fps,
        "near": parser.near,
        "far": parser.far,
        "lookat": parser.lookat.tolist(),
        "up": parser.up.tolist(),
    }
    io.dump(osp.join(dump_dir, "extra.json"), extra_dict)


if __name__ == "__main__":
    app.run(main)
