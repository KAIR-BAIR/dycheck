#!/usr/bin/env python3
#
# File   : trajs.py
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

from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Camera
from .utils import matv


def get_arc_traj(
    ref_camera: Camera,
    lookat: np.ndarray,
    up: np.ndarray,
    *,
    num_frames: int,
    degree: float,
    **_,
) -> List[Camera]:
    positions = [
        matv(
            Rotation.from_rotvec(d / 180 * np.pi * up).as_matrix()
            @ (ref_camera.position - lookat)
        )
        + lookat
        for d in np.linspace(-degree / 2, degree / 2, num_frames)
    ]
    cameras = [ref_camera.lookat(p, lookat, up) for p in positions]
    return cameras


def get_lemniscate_traj(
    ref_camera: Camera,
    lookat: np.ndarray,
    up: np.ndarray,
    *,
    num_frames: int,
    degree: float,
    **_,
) -> List[Camera]:
    a = np.linalg.norm(ref_camera.position - lookat) * np.tan(
        degree / 360 * np.pi
    )
    # Lemniscate curve in camera space. Starting at the origin.
    positions = np.array(
        [
            np.array(
                [
                    a * np.cos(t) / (1 + np.sin(t) ** 2),
                    a * np.cos(t) * np.sin(t) / (1 + np.sin(t) ** 2),
                    0,
                ]
            )
            for t in (np.linspace(0, 2 * np.pi, num_frames) + np.pi / 2)
        ]
    )
    # Transform to world space.
    positions = matv(ref_camera.orientation.T, positions) + ref_camera.position
    cameras = [ref_camera.lookat(p, lookat, up) for p in positions]
    return cameras
