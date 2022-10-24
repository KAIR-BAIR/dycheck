#!/usr/bin/env python3
#
# File   : skeleton.py
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

import re
from typing import Callable, Optional, Sequence, Tuple, Union

import gin
import numpy as np
from matplotlib import cm

from dycheck.utils import image

KP_PALETTE_MAP = {}


@gin.configurable()
class Skeleton(object):
    name = "skeleton"
    _anonymous_kp_name = "ANONYMOUS KP"

    def __init__(
        self,
        parents: Sequence[Optional[int]],
        kp_names: Optional[Sequence[str]] = None,
        palette: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        if kp_names is not None:
            assert len(parents) == len(kp_names)
            if palette is not None:
                assert len(kp_names) == len(palette)

        self._parents = parents
        self._kp_names = (
            kp_names
            if kp_names is not None
            else [self._anonymous_kp_name] * self.num_kps
        )
        self._palette = palette

    def asdict(self):
        return {
            "name": self.name,
            "parents": self.parents,
            "kp_names": self.kp_names,
            "palette": self.palette,
        }

    @property
    def is_unconnected(self):
        return all([p is None for p in self._parents])

    @property
    def parents(self):
        return self._parents

    @property
    def kp_names(self):
        return self._kp_names

    @property
    def palette(self):
        if self._palette is not None:
            return self._palette

        if self.kp_names[0] != self._anonymous_kp_name and all(
            [kp_name in KP_PALETTE_MAP for kp_name in self.kp_names]
        ):
            return [KP_PALETTE_MAP[kp_name] for kp_name in self.kp_names]

        palette = np.zeros((self.num_kps, 3), dtype=np.uint8)
        left_mask = np.array(
            [
                len(re.findall(r"^(\w+ |)L\w+$", kp_name)) > 0
                for kp_name in self._kp_names
            ],
            dtype=np.bool,
        )
        palette[left_mask] = (255, 0, 0)
        return [tuple(color.tolist()) for color in palette]

    @property
    def num_kps(self):
        return len(self._parents)

    @property
    def root_idx(self):
        if self.is_unconnected:
            return 0
        return self._parents.index(-1)

    @property
    def bones(self):
        if self.is_unconnected:
            return []
        return np.stack([list(range(self.num_kps)), self.parents]).T.tolist()

    @property
    def non_root_bones(self):
        if self.is_unconnected:
            return []
        return np.delete(self.bones.copy(), self.root_idx, axis=0)

    @property
    def non_root_palette(self):
        if self.is_unconnected:
            return []
        return np.delete(self.palette.copy(), self.root_idx, axis=0).tolist()


@gin.configurable()
class UnconnectedSkeleton(Skeleton):
    """A keypoint skeleton that does not define parents. This could be useful
    when organizing randomly annotated keypoints.
    """

    name: str = "unconnected"

    def __init__(
        self, num_kps: int, cmap: Union[str, Callable] = "gist_rainbow"
    ):
        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap, num_kps)
        pallete = image.to_uint8(
            np.array([cmap(i)[:3] for i in range(num_kps)], np.float32)
        ).tolist()
        super().__init__(
            parents=[None for _ in range(num_kps)],
            kp_names=[f"KP_{i}" for i in range(num_kps)],
            palette=pallete,
        )


@gin.configurable()
class HumanSkeleton(Skeleton):
    """A human skeleton following the COCO dataset.

    Microsoft COCO: Common Objects in Context.
        Lin et al., ECCV 2014.
        https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48

    For pictorial definition, see also: shorturl.at/ilnpZ.
    """

    name: str = "human"

    def __init__(self, **_):
        super().__init__(
            parents=[
                1,
                -1,
                1,
                2,
                3,
                1,
                5,
                6,
                1,
                8,
                9,
                1,
                11,
                12,
                0,
                0,
                14,
                15,
            ],
            kp_names=[
                "Nose",
                "Neck",
                "RShoulder",
                "RElbow",
                "RWrist",
                "LShoulder",
                "LElbow",
                "LWrist",
                "RHip",
                "RKnee",
                "RAnkle",
                "LHip",
                "LKnee",
                "LAnkle",
                "REye",
                "LEye",
                "REar",
                "LEar",
            ],
            palette=[
                (255, 0, 0),
                (255, 85, 0),
                (255, 170, 0),
                (255, 255, 0),
                (170, 255, 0),
                (85, 255, 0),
                (0, 255, 0),
                (0, 255, 85),
                (0, 255, 170),
                (0, 255, 255),
                (0, 170, 255),
                (0, 85, 255),
                (0, 0, 255),
                (85, 0, 255),
                (170, 0, 255),
                (255, 0, 255),
                (255, 0, 170),
                (255, 0, 85),
            ],
        )


@gin.configurable()
class QuadrupedSkeleton(Skeleton):
    """A quadruped skeleton following StanfordExtra dataset.

    Novel dataset for Fine-Grained Image Categorization.
        Khosla et al., CVPR 2011, FGVC workshop.
        http://vision.stanford.edu/aditya86/ImageNetDogs/main.html

    Who Left the Dogs Out? 3D Animal Reconstruction with Expectation
    Maximization in the Loop.
        Biggs et al., ECCV 2020.
        https://arxiv.org/abs/2007.11110
    """

    name: str = "quadruped"

    def __init__(self, **_):
        super().__init__(
            parents=[
                1,
                2,
                22,
                4,
                5,
                12,
                7,
                8,
                22,
                10,
                11,
                12,
                -1,
                12,
                20,
                21,
                17,
                23,
                14,
                15,
                16,
                16,
                12,
                22,
            ],
            kp_names=[
                "LFrontPaw",
                "LFrontWrist",
                "LFrontElbow",
                "LRearPaw",
                "LRearWrist",
                "LRearElbow",
                "RFrontPaw",
                "RFrontWrist",
                "RFrontElbow",
                "RRearPaw",
                "RRearWrist",
                "RRearElbow",
                "TailStart",
                "TailEnd",
                "LEar",
                "REar",
                "Nose",
                "Chin",
                "LEarTip",
                "REarTip",
                "LEye",
                "REye",
                "Withers",
                "Throat",
            ],
            palette=[
                (0, 255, 0),
                (63, 255, 0),
                (127, 255, 0),
                (0, 0, 255),
                (0, 63, 255),
                (0, 127, 255),
                (255, 255, 0),
                (255, 191, 0),
                (255, 127, 0),
                (0, 255, 255),
                (0, 255, 191),
                (0, 255, 127),
                (0, 0, 0),
                (0, 0, 0),
                (255, 0, 170),
                (255, 0, 170),
                (255, 0, 170),
                (255, 0, 170),
                (255, 0, 170),
                (255, 0, 170),
                (255, 0, 170),
                (255, 0, 170),
                (255, 0, 170),
                (255, 0, 170),
            ],
        )


SKELETON_MAP = {
    cls.name: cls
    for cls in [
        Skeleton,
        UnconnectedSkeleton,
        HumanSkeleton,
        QuadrupedSkeleton,
    ]
}
