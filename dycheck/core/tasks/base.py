#!/usr/bin/env python3
#
# File   : base.py
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

from typing import Optional, Sequence

import gin

from dycheck.utils import types


class Task(object):
    def __init__(
        self,
        engine: types.EngineType,
        *,
        interval: Optional[int] = None,
    ):
        self.engine = engine
        self.interval = interval

    @property
    def eligible(self):
        return True

    def start(self):
        raise NotImplementedError

    def every_n_steps(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError


@gin.configurable(denylist=["engine"])
class Tasks(object):
    def __init__(
        self,
        engine: types.EngineType,
        task_classes: Sequence[Task] = gin.REQUIRED,
    ):
        self.engine = engine
        self.tasks = [task_cls(engine) for task_cls in task_classes]

    def start(self):
        for t in self.tasks:
            if t.eligible:
                t.start()

    def every_n_steps(self):
        engine = self.engine

        for t in self.tasks:
            if (
                engine.training
                and t.eligible
                and t.interval is not None
                and engine.step % t.interval == 0
                # Assume trainer always has a max step.
                and engine.step != engine.max_steps
            ):
                t.every_n_steps()

    def finalize(self):
        for t in self.tasks:
            if t.eligible:
                t.finalize()
