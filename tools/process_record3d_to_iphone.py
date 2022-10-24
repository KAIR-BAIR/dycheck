#!/usr/bin/env python3
#
# File   : process_record3d_to_iphone.py
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

import gin
from absl import app, flags, logging

from dycheck import core
from dycheck.datasets import Record3DProcessor

flags.DEFINE_multi_string(
    "gin_configs", None, "Gin config files.", required=True
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
FLAGS = flags.FLAGS


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

    logging.info("*** Starting processing Record3D to iPhone data format.")
    processor = Record3DProcessor()
    processor.process()


if __name__ == "__main__":
    app.run(main)
