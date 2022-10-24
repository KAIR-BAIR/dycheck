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

import os.path as osp
import queue
import threading
from typing import Dict, Optional, Tuple

import gin
import jax
import numpy as np

from dycheck import geometry
from dycheck.nn import functional as F
from dycheck.utils import types, visuals


class MetadataInfoMixin(object):
    @property
    def frame_names(self):
        raise NotImplementedError

    @property
    def time_ids(self):
        raise NotImplementedError

    @property
    def camera_ids(self):
        raise NotImplementedError

    @property
    def uniq_time_ids(self):
        return np.unique(self.time_ids)

    @property
    def uniq_camera_ids(self):
        return np.unique(self.camera_ids)

    @property
    def num_frames(self):
        return len(self.frame_names)

    @property
    def num_times(self):
        return len(set(self.time_ids))

    @property
    def num_cameras(self):
        return len(set(self.camera_ids))

    @property
    def embeddings_dict(self):
        return {"time": self.uniq_time_ids, "camera": self.uniq_camera_ids}

    def validate_metadata_info(self):
        if not (np.ediff1d(self.uniq_time_ids) == 1).all():
            raise ValueError("Unique time ids are not consecutive.")
        if not (np.ediff1d(self.uniq_camera_ids) == 1).all():
            raise ValueError("Unique camera ids are not consecutive.")


class Parser(MetadataInfoMixin, object):
    """Parser for parsing and loading raw data without any preprocessing or
    data splitting.
    """

    def __init__(
        self,
        dataset: str,
        sequence: str,
        *,
        data_root: Optional[types.PathType] = None,
    ):
        self.dataset = dataset
        self.sequence = sequence
        self.data_root = data_root or osp.abspath(
            osp.join(osp.dirname(__file__), "..", "..", "datasets")
        )
        self.data_dir = osp.join(self.data_root, self.dataset, self.sequence)

    def load_rgba(self, time_id: int, camera_id: int) -> np.ndarray:
        raise NotImplementedError(
            f"Load RGBA with time_id={time_id}, camera_id={camera_id}."
        )

    def load_depth(self, time_id: int, camera_id: int) -> np.ndarray:
        raise NotImplementedError(
            f"Load depth with time_id={time_id}, camera_id={camera_id}."
        )

    def load_camera(
        self, time_id: int, camera_id: int, **_
    ) -> geometry.Camera:
        raise NotImplementedError(
            f"Load camera with time_id={time_id}, camera_id={camera_id}."
        )

    def load_covisible(
        self, time_id: int, camera_id: int, split: str, **_
    ) -> np.ndarray:
        raise NotImplementedError(
            f"Load covisible with time_id={time_id}, "
            f"camera_id={camera_id}, split={split}."
        )

    def load_keypoints(
        self, time_id: int, camera_id: int, split: str, **_
    ) -> np.ndarray:
        raise NotImplementedError(
            f"Load keypoints with time_id={time_id}, "
            f"camera_id={camera_id}, split={split}."
        )

    def load_skeleton(self, split: str) -> visuals.Skeleton:
        raise NotImplementedError(f"Load skeleton with split={split}.")

    def load_split(
        self, split: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError(f"Load split {split}.")

    @property
    def center(self):
        raise NotImplementedError

    @property
    def scale(self):
        raise NotImplementedError

    @property
    def near(self):
        raise NotImplementedError

    @property
    def far(self):
        raise NotImplementedError

    @property
    def factor(self):
        raise NotImplementedError

    @property
    def fps(self):
        raise NotImplementedError

    @property
    def bbox(self):
        raise NotImplementedError

    @property
    def lookat(self):
        raise NotImplementedError

    @property
    def up(self):
        raise NotImplementedError


class Dataset(threading.Thread, MetadataInfoMixin):
    """Cached dataset for training and evaluation."""

    __parser_cls__: Parser = None

    def __init__(
        self,
        dataset: str = gin.REQUIRED,
        sequence: str = gin.REQUIRED,
        batch_size: int = 0,
        *,
        split: Optional[str] = None,
        training: Optional[bool] = None,
        cache_num_repeat: int = 1,
        seed: int = 0,
        bkgd_points_batch_size: int = 0,
        **_,
    ):
        super().__init__(daemon=True)
        self._queue = queue.Queue(3)
        self._num_repeats = 0
        self._index = 0

        self.dataset = dataset
        self.sequence = sequence
        self.batch_size = batch_size

        self.split = split
        if training is None:
            training = self.split is not None and self.split.startswith(
                "train"
            )
        self.training = training
        self.cache_num_repeat = cache_num_repeat
        self.seed = seed
        self.bkgd_points_batch_size = bkgd_points_batch_size

        # RandomState's choice method is too slow.
        self.rng = np.random.default_rng(seed)

        if self.training:
            device_count = jax.local_device_count()
            assert self.batch_size > 0 and self.batch_size % device_count == 0
            if self.bkgd_points_batch_size > 0:
                assert self.bkgd_points_batch_size % device_count == 0
        else:
            self.batch_size = 0
            self.bkgd_points_batch_size = 0

        assert self.__parser_cls__, "Parser class is not defined."
        self.parser = self.__parser_cls__(self.dataset, self.sequence)

    def fetch_data(self, index: int):
        """Fetch the data (it maybe cached for multiple batches)."""
        raise NotImplementedError(f"Fetch data at index {index}.")

    def preprocess(self, data: Dict):
        """Process the fetched / cached data with randomness.

        Note that in-place operations should be avoided since the raw data
        might be repeated for serveral times with different randomness.
        """
        raise NotImplementedError(f"Preprocess data {data}.")

    def __iter__(self):
        return self

    def __next__(self):
        """Fetch cached data and preprocess."""
        if self.training and self._num_repeats < self.cache_num_repeat:
            data = self._queue.queue[0]
            self._num_repeats += 1
        else:
            data = self._queue.get()
            self._num_repeats = 1
        batch = self.preprocess(data)

        return F.shard(batch) if self.training else F.to_device(batch)

    def run(self):
        """Main data fetching loop of the thread."""
        while True:
            if self.training:
                index = np.random.randint(0, len(self))
            else:
                index = self._index
                self._index = self._index + 1
                self._index %= len(self)

            data = self.fetch_data(index)
            self._queue.put(data)

    def __getitem__(self, index: int):
        """Seek the batch by index. Only for testing purpose."""
        data = self.fetch_data(index)
        return self.preprocess(data)

    @classmethod
    def create(cls, *args, **kwargs):
        """A wrapper around __init__ such that always start fetching *after*
        subclasses get initialized.

        Note that __post_init__ does not work in this case.
        """
        dataset = cls(*args, **kwargs)
        dataset.start()
        return dataset

    @classmethod
    def create_dummy(cls, *args, **kwargs):
        """Create dummy dataset such that no prefetching is performed.

        This method can be useful when evaluating.
        """
        dataset = cls(*args, **kwargs)
        return dataset

    @property
    def data_dir(self):
        return self.parser.data_dir

    @property
    def has_novel_view(self):
        raise NotImplementedError

    @property
    def has_keypoints(self):
        raise NotImplementedError

    @property
    def center(self):
        return self.parser.center

    @property
    def scale(self):
        return self.parser.scale

    @property
    def near(self):
        return self.parser.near

    @property
    def far(self):
        return self.parser.far

    @property
    def factor(self):
        return self.parser.factor

    @property
    def fps(self):
        return self.parser.fps

    @property
    def bbox(self):
        return self.parser.bbox

    @property
    def lookat(self):
        return self.parser.lookat

    @property
    def up(self):
        return self.parser.up

    def __len__(self):
        return self.num_frames
