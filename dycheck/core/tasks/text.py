#!/usr/bin/env python3
#
# File   : text.py
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
import time
from datetime import timedelta
from string import Formatter

import gin
import jax
import numpy as np
from absl import logging
from flax import traverse_util

from . import base


@gin.configurable(denylist=["engine"])
class Text(base.Task):
    """Logging text to stdout periodically during training."""

    @property
    def eligible(self):
        return self.engine.training

    def start(self):
        pass

    def every_n_steps(self):
        engine = self.engine

        stats = {}
        steps_per_sec = self.interval / (time.time() - engine.start_time)
        rays_per_sec = engine.dataset.batch_size * steps_per_sec
        stats["time"] = {
            "steps_per_sec": steps_per_sec,
            "rays_per_sec": rays_per_sec,
        }
        stats["param"] = {
            "step": engine.step,
            **dataclasses.asdict(engine.scalars),
            **engine.extra_params._asdict(),
        }
        _stats = jax.tree_map(lambda x: x.mean().item(), engine.pstats)
        stats.update(
            **traverse_util.unflatten_dict(
                {tuple(k.split("/")): v for k, v in _stats.items()}
            )
        )
        engine.stats = stats

        # Logging to screen.
        precision = int(np.ceil(np.log10(engine.max_steps))) + 1
        eta = timedelta(
            seconds=(engine.max_steps - engine.step) / steps_per_sec
        )
        text_str = ("{:" + "{:d}".format(precision) + "d}").format(
            engine.step
        ) + f"/{engine.max_steps:d} ({strfdelta(eta)})"
        text_str += ".\n    time: " + ", ".join(
            [f"{k}={v:0.2e}" for k, v in stats["time"].items()]
        )
        text_str += ".\n    param: " + ", ".join(
            [f"{k}={v:0.4f}" for k, v in stats["param"].items() if k != "step"]
        )
        text_str += ".\n    loss: " + ", ".join(
            [f"{k}={v:0.4f}" for k, v in stats.get("loss", {}).items()]
        )
        text_str += (
            ".\n    metric: "
            + ", ".join(
                [f"{k}={v:0.4f}" for k, v in stats.get("metric", {}).items()]
            )
            + "."
        )
        logging.info(text_str)

        # Logging to tensorboard.
        summary_stats = {
            "/".join(k): v
            for k, v in traverse_util.flatten_dict(stats).items()
        }
        for k, v in summary_stats.items():
            engine.summary_writer.scalar(k, v, engine.step)

        # Reset timer.
        engine.start_time = time.time()

    def finalize(self):
        pass


def strfdelta(
    tdelta: timedelta,
    fmt: str = "{D:02}d {H:02}h {M:02}m {S:02}s",
    inputtype: str = "timedelta",
):
    """Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The inputtype argument allows tdelta to be a regular number instead of the
    default, which is a datetime.timedelta object.  Valid inputtype strings:
        's', 'seconds',
        'm', 'minutes',
        'h', 'hours',
        'd', 'days',
        'w', 'weeks'
    """

    # Convert tdelta to integer seconds.
    if inputtype == "timedelta":
        remainder = int(tdelta.total_seconds())
    elif inputtype in ["s", "seconds"]:
        remainder = int(tdelta)
    elif inputtype in ["m", "minutes"]:
        remainder = int(tdelta) * 60
    elif inputtype in ["h", "hours"]:
        remainder = int(tdelta) * 3600
    elif inputtype in ["d", "days"]:
        remainder = int(tdelta) * 86400
    elif inputtype in ["w", "weeks"]:
        remainder = int(tdelta) * 604800
    else:
        raise ValueError(inputtype)

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ("W", "D", "H", "M", "S")
    constants = {"W": 604800, "D": 86400, "H": 3600, "M": 60, "S": 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)
