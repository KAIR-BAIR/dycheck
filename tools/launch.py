#!/usr/bin/env python3
#
# File   : launch.sh
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

import os
import warnings

# Filter out false positive warnings like threading timeout from jax and
# tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import dataclasses

import gin
import numpy as np
import tensorflow as tf
from absl import app, flags, logging

from dycheck import core

flags.DEFINE_multi_string(
    "gin_configs", None, "Gin config files.", required=True
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
FLAGS = flags.FLAGS


def main(_):
    # Make sure that tensorflow does not allocate any GPU memory. JAX will
    # somehow use tensorflow internally and without this line data loading is
    # likely to result in OOM.
    tf.config.experimental.set_visible_devices([], "GPU")
    np.random.seed(0)

    logging.info(f"*** Loading Gin configs from: {FLAGS.gin_configs}.")
    core.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs,
        bindings=FLAGS.gin_bindings,
        skip_unknown=True,
    )

    config_str = gin.config_str()
    logging.info(f"*** Configuration:\n{config_str}")

    config = core.Config()
    engine = config.engine_cls(
        **{
            k: v
            for k, v in dataclasses.asdict(config).items()
            if k != "engine_cls"
        }
    )
    logging.info(
        f"*** Starting experiment:\n"
        f"    name={engine.name}\n"
        f"    data_dir={engine.dataset.data_dir}\n"
        f"    work_dir={engine.work_dir}"
    )
    engine.launch()


if __name__ == "__main__":
    app.run(main)
