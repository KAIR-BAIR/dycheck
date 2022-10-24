#!/usr/bin/env python3
#
# File   : launch.py
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

import os.path as osp

import gin
import jax.numpy as jnp
import tensorflow as tf
from absl import app, flags, logging
from flax import jax_utils, optim
from flax.training import checkpoints
from jax import random

from dycheck import core, geometry
from dycheck.core.tasks.video import Video
from dycheck.utils import common, io, struct

flags.DEFINE_multi_string(
    "gin_configs",
    "configs/iphone/tnerf/randbkgd_depth_dist.gin",
    "Gin config files.",
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
flags.DEFINE_enum(
    "task",
    "novel_view",
    ["novel_view", "stabilized_view", "bullet_time"],
    "The task to run.",
)
FLAGS = flags.FLAGS

DATA_ROOT = osp.join(osp.dirname(__file__), "datasets")
WORK_ROOT = osp.join(osp.dirname(__file__), "work_dirs")
FPS = 20


def main(_):
    # Make sure that tensorflow does not allocate any GPU memory. JAX will
    # somehow use tensorflow internally and without this line data loading is
    # likely to result in OOM.
    tf.config.experimental.set_visible_devices([], "GPU")

    core.parse_config_files_and_bindings(
        # Hard-code for demo.
        config_files=FLAGS.gin_configs,
        bindings=(FLAGS.gin_bindings or [])
        + [
            # Engine is not used for demo.
            "Config.engine_cls=None",
            "SEQUENCE='paper-windmill'",
            f"Config.work_root='{WORK_ROOT}'",
            f"iPhoneParser.data_root='{DATA_ROOT}'",
            f"iPhoneDatasetFromAllFrames.split='train'",
            f"iPhoneDatasetFromAllFrames.training=False",
            f"iPhoneDatasetFromAllFrames.bkgd_points_batch_size=0",
        ],
        skip_unknown=True,
    )

    config_str = gin.config_str()
    logging.info(f"*** Configuration:\n{config_str}")

    config = core.Config()

    # Create dataset.
    dataset = config.dataset_cls()
    # Setup paths.
    work_dir = osp.join(
        config.work_root,
        dataset.dataset,
        dataset.sequence,
        config.name,
    )
    checkpoint_dir = osp.join(work_dir, "checkpoints")
    renders_dir = osp.join(work_dir, "renders")
    # Load training camera infos. These cameras are going to be used for
    # ``culling'' the unseen part of the scene.
    train_cameras = common.parallel_map(
        dataset.parser.load_camera,
        dataset.time_ids,
        dataset.camera_ids,
    )
    train_cameras_dict = common.tree_collate(
        [
            {
                "intrin": c.intrin,
                "extrin": c.extrin,
                "radial_distortion": c.radial_distortion,
                "tangential_distortion": c.tangential_distortion,
                "image_size": c.image_size,
            }
            for c in train_cameras
        ],
        collate_fn=lambda *x: jnp.array(x),
    )
    # Create model.
    key = random.PRNGKey(0)
    model, variables = config.model_cls.create(
        key=key,
        embeddings_dict=dataset.embeddings_dict,
        cameras_dict=train_cameras_dict,
        near=dataset.near,
        far=dataset.far,
    )
    optimizer = optim.Adam(0).create(variables)
    state = checkpoints.restore_checkpoint(
        checkpoint_dir,
        struct.TrainState(optimizer=optimizer),
    )
    pstate = jax_utils.replicate(state)

    # In this demo, we are going to re-render a video given a `FLAGS.task`.
    # 1. FLAGS.task == "novel_view":
    #   Render a novel-view video around the camera pose of the first frame
    #   while fix the motion, similar to Nerfies/HyperNeRF.
    # 2. FLAGS.task == "stabilized_view":
    #   Render a ``stabilized'' video by fixing the camera pose of the first
    #   frame while replaying the whole video, similar to NSFF.
    # 3. FLAGS.task == "bullet_time":
    #   Render a ``bullet-time'' video by moving the camera around the first
    #   frame's pose while replaying the whole video, similar to NSFF.
    task = FLAGS.task
    camera = train_cameras[0]
    if task != "stabilized_view":
        cameras = geometry.get_lemniscate_traj(
            camera,
            lookat=dataset.lookat,
            up=dataset.up,
            num_frames=60,
            degree=30,
        )
    else:
        cameras = [camera]
    image_shape = tuple(cameras[0].image_shape)
    if task != "novel_view":
        time_ids = dataset.time_ids.tolist()
    else:
        time_ids = [dataset.time_ids[0]]
    camera_ids = [dataset.camera_ids[0] for _ in range(len(time_ids))]
    metadatas = [
        struct.Metadata(
            time=jnp.full(image_shape + (1,), t),
            camera=jnp.full(image_shape + (1,), c),
        )
        for t, c in zip(time_ids, camera_ids)
    ]
    # Pad the cameras and metadatas to the same length.
    cameras, metadatas = Video.pad_by_fps(
        cameras,
        metadatas,
        dataset_fps=dataset.fps,
        target_fps=FPS if task != "stabilized_view" else dataset.fps,
    )

    # Render the video.
    prender_image = core.get_prender_image(model, use_cull_cameras=True)
    video = []
    for camera, metadata in zip(
        common.tqdm(
            cameras,
            desc=f"* Rendering demo novel-view video",
            position=1,
            leave=False,
        ),
        metadatas,
    ):
        rays = camera.pixels_to_rays(camera.get_pixels())._replace(
            metadata=metadata
        )
        rendered = prender_image(
            pstate.optimizer.target,
            rays,
            key=key,
            show_pbar=False,
        )
        video.append(rendered["rgb"])
    video_path = osp.join(renders_dir, f"{task}.mp4")
    io.dump(
        video_path,
        video,
        fps=FPS,
        show_pbar=False,
    )
    logging.info(f"* Videos dumped at {video_path}.")


if __name__ == "__main__":
    app.run(main)
