#!/usr/bin/env python3
#
# File   : combine_record3d_av.sh
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
import subprocess

import gin
from absl import app, flags, logging

from dycheck import core
from dycheck.utils import path_ops, types

flags.DEFINE_multi_string(
    "gin_configs", None, "Gin config files.", required=True
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
FLAGS = flags.FLAGS


@gin.configurable(module="combine_record3d_av")
@dataclasses.dataclass
class Config(object):
    sequence: str = gin.REQUIRED
    data_root: types.PathType = osp.join(
        "/shared/hangg/datasets/iphone/record3d/"
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

    logging.info("*** Starting combining Record3D audio and video.")

    data_dir = osp.join(config.data_root, config.sequence)

    def combine_av(root_dir):
        video_paths = path_ops.ls(osp.join(root_dir, "RGBD_Video/*.mp4"))
        assert len(video_paths) == 1
        video_path = video_paths[0]
        new_video_path = video_path + ".mp4"

        audio_path = osp.join(root_dir, "sound.m4a")

        # FFMPEG cannot work inplace.
        path_ops.cp(video_path, new_video_path)
        try:
            subprocess.run(
                (
                    f"ffmpeg -i {new_video_path} -i {audio_path} "
                    f"-acodec copy -vcodec copy {video_path} -y"
                ),
                shell=True,
                check=True,
            )
        finally:
            path_ops.rm(new_video_path)

    combine_av(data_dir)
    for test_dir in path_ops.ls(osp.join(data_dir, "Test/*"), type="d"):
        combine_av(test_dir)


if __name__ == "__main__":
    app.run(main)
