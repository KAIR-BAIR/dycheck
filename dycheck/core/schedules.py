#!/usr/bin/env python3
#
# File   : schedules.py
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

import math
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import gin
import numpy as np


class Schedule(object):
    """An interface for generic schedules.."""

    def get(self, step):
        """Get the value for the given step."""
        raise NotImplementedError(f"Get {step}.")

    def __call__(self, step):
        return self.get(step)


@gin.configurable()
class ConstantSchedule(Schedule):
    """Constant scheduler."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def get(self, _: int):
        """Get the value for the given step."""
        return self.value


@gin.configurable()
class LinearSchedule(Schedule):
    """Linearly scaled scheduler."""

    def __init__(
        self, initial_value: float, final_value: float, num_steps: int
    ):
        super().__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps

    def get(self, step: int):
        """Get the value for the given step."""
        if self.num_steps == 0:
            return self.final_value
        alpha = min(step / self.num_steps, 1.0)
        return (1.0 - alpha) * self.initial_value + alpha * self.final_value


@gin.configurable()
class ExponentialSchedule(Schedule):
    """Exponentially decaying scheduler."""

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        num_steps: int,
        eps: float = 1e-10,
    ):
        super().__init__()
        if initial_value <= final_value:
            raise ValueError("Final value must be less than initial value.")

        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps
        self.eps = eps

    def get(self, step: int):
        """Get the value for the given step."""
        if step >= self.num_steps:
            return self.final_value

        final_value = max(self.final_value, self.eps)
        base = final_value / self.initial_value
        exponent = step / (self.num_steps - 1)
        return self.initial_value * base**exponent


@gin.configurable()
class CosineEasingSchedule(Schedule):
    """A scheduler that eases slowsly using a cosine."""

    def __init__(
        self, initial_value: float, final_value: float, num_steps: int
    ):
        super().__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps

    def get(self, step: int):
        """Get the value for the given step."""
        alpha = min(step / self.num_steps, 1.0)
        scale = self.final_value - self.initial_value
        x = min(max(alpha, 0.0), 1.0)
        return self.initial_value + scale * 0.5 * (
            1 + math.cos(np.pi * x + np.pi)
        )


@gin.configurable()
class StepSchedule(Schedule):
    """Step decaying scheduler."""

    def __init__(
        self,
        initial_value: float,
        decay_interval: int,
        decay_factor: float,
        max_decays: int,
        final_value: Optional[float] = None,
    ):
        super().__init__()
        self.initial_value = initial_value
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.max_decays = max_decays
        if final_value is None:
            final_value = (
                self.initial_value * self.decay_factor**self.max_decays
            )
        self.final_value = final_value

    def get(self, step: int):
        """Get the value for the given step."""
        phase = step // self.decay_interval
        if phase >= self.max_decays:
            return self.final_value
        else:
            return self.initial_value * self.decay_factor**phase


@gin.configurable()
class PiecewiseSchedule(Schedule):
    """A piecewise combination of multiple schedules."""

    def __init__(
        self, schedules: Iterable[Tuple[int, Union[Schedule, Iterable[Any]]]]
    ):
        self.schedules = [s for _, s in schedules]
        milestones = np.array([ms for ms, _ in schedules])
        self.milestones = np.cumsum(milestones)[:-1]

    def get(self, step):
        idx = np.searchsorted(self.milestones, step, side="right")
        schedule = self.schedules[idx]
        base_idx = self.milestones[idx - 1] if idx >= 1 else 0
        return schedule.get(step - base_idx)


@gin.configurable()
class WarmupExponentialSchedule(Schedule):
    """Exponentially decaying scheduler combined with a warmup initialization.

    This scheduler matches the one in jaxNerf.
    """

    eps = 1e-10

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        num_steps: int,
        lr_delay_steps: int = 0,
        lr_delay_mult: float = 1,
    ):
        super().__init__()
        final_value = max(final_value, self.eps)
        initial_value = max(final_value, initial_value)

        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult

    def get(self, step):
        if step >= self.num_steps:
            return self.final_value

        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (
                1 - self.lr_delay_mult
            ) * np.sin(0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.0
        t = np.clip(step / self.num_steps, 0, 1)
        log_lerp = np.exp(
            np.log(self.initial_value) * (1 - t) + np.log(self.final_value) * t
        )
        return delay_rate * log_lerp


@gin.configurable()
class ZipSchedule(Schedule):
    """A scheduler that zips values from other schedulers"""

    def __init__(
        self, schedules: Iterable[Union[Schedule, Dict[str, Any], Tuple]]
    ):
        self.schedules = [s for s in schedules]

    def get(self, step):
        return tuple(s(step) for s in self.schedules)
