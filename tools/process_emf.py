#!/usr/bin/env python3
#
# File   : process_emf.py
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
from collections import defaultdict
from typing import Callable, Sequence

import gin
import numpy as np
from absl import app, flags, logging

from dycheck import core, processors
from dycheck.core.tasks import utils
from dycheck.datasets import Parser
from dycheck.utils import common, io

flags.DEFINE_multi_string(
    "gin_configs", None, "Gin config files.", required=True
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
FLAGS = flags.FLAGS


@gin.configurable(module="process_emf")
@dataclasses.dataclass
class Config(object):
    parser_cls: Callable[..., Parser] = gin.REQUIRED
    splits: Sequence[str] = gin.REQUIRED


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

    logging.info("*** Starting processing effective multi-view factors.")
    parser = config.parser_cls()
    assert not getattr(
        parser, "use_undistort", False
    ), "Currently only support undistorted images for EMF computation."

    emf_path = osp.join(parser.data_dir, "emf.json")

    emf_dict = defaultdict(dict)
    for split in config.splits:
        _, time_ids, camera_ids = parser.load_split(split)
        rgbs = np.array(
            common.parallel_map(parser.load_rgba, time_ids, camera_ids)
        )[..., :3]
        cameras = common.parallel_map(parser.load_camera, time_ids, camera_ids)

        # 1. Angular EMF omega: camera angular speed.
        emf_dict[split]["omega"] = processors.compute_angular_emf(
            np.stack([c.orientation for c in cameras], axis=0),
            np.stack([c.position for c in cameras], axis=0),
            fps=parser.fps,
            lookat=parser.lookat,
        )

        # 2. Full EMF Omega: relative camera-scene motion ratio.
        emf_dict[split]["Omega"] = processors.compute_full_emf(
            rgbs,
            cameras,
            parser.load_bkgd_points().astype(np.float32),
        )

        logging.info(
            (
                f"* EMF statistics ({split}):\n"
                f"{utils.format_dict(emf_dict[split])}"
            )
        )

    io.dump(emf_path, emf_dict)


if __name__ == "__main__":
    app.run(main)
