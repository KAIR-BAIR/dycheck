#!/usr/bin/env python3
#
# File   : config.py
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
import logging
import os.path as osp
from typing import Callable, Optional, Sequence

import gin
from flax import linen as nn

from dycheck.datasets import Dataset
from dycheck.utils import types

from .engines import Engine

gin.add_config_file_search_path(
    osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "configs"))
)


def parse_config_files_and_bindings(
    config_files: Sequence[types.PathType],
    bindings: Sequence[str],
    skip_unknown: bool = True,
    refresh: bool = True,
    master: bool = True,
    **kwargs,
):
    """Parse config files and bindings.

    Args:
        config_files (Sequence[PathType]): Paths to config files.
        bindings (Sequence[str]): Gin parameter bindings.
        skip_unknown (bool): Whether to skip unknown parameters.
        refresh (bool): Whether to refresh the config.
    """
    if refresh:
        gin.clear_config()

    # Parse config files. Filter out false positive warnings.
    logging.disable(level=logging.CRITICAL)
    _ = gin.parse_config_files_and_bindings(
        config_files=config_files,
        bindings=bindings,
        skip_unknown=skip_unknown,
        **kwargs,
    )
    logging.disable(level=logging.NOTSET)

    if master:
        # Hardcode value evaluations for master config.
        config = Config()
        name = config.name
        dataset = gin.config._CONFIG[("DATASET", "gin.macro")]["value"]
        if name is None:
            # Hardcode the name of the engine by config file name if not
            # specified.
            name = config_files[0]
            prefix = f"configs/{dataset}/"
            name = name[name.rfind(prefix) + len(prefix) :]
            name = name[: name.rfind(".gin")]
            gin.config._CONFIG[("", "dycheck.core.config.Config")][
                "name"
            ] = name


@gin.configurable()
@dataclasses.dataclass
class Config(object):
    engine_cls: Callable[..., Engine] = gin.REQUIRED
    dataset_cls: Callable[..., Dataset] = gin.REQUIRED
    model_cls: Callable[..., nn.Module] = gin.REQUIRED
    name: Optional[str] = None
    work_root: Optional[types.PathType] = None
    checkpoint_step: Optional[int] = None
    checkpoint_path: Optional[types.PathType] = None
