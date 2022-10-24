#!/usr/bin/env python3
#
#
# File   : iphone.py
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

import functools
import os.path as osp
from typing import Literal, Optional, Sequence
from zipfile import ZipFile

import cv2
import gin
import numpy as np
import trimesh
from absl import logging
from scipy.spatial.transform import Rotation

from dycheck import geometry, processors
from dycheck.utils import common, image, io, path_ops, types, visuals

from . import utils
from .base import Parser
from .nerfies import (
    NerfiesDatasetFromAllFrames,
    NerfiesParser,
    _load_metadata_info,
    _load_scene_info,
)

SPLITS = [
    "train",
    "val",
]

DEFAULT_FACTOR: int = 2


@gin.configurable()
class Record3DProcessor(object):
    def __init__(
        self,
        sequence: str = gin.REQUIRED,
        frgd_prompts: Sequence[str] = gin.REQUIRED,
        *,
        data_root: Optional[types.PathType] = None,
        dump_root: Optional[types.PathType] = None,
        rotate_mode: Literal[
            "clockwise_0", "clockwise_90", "clockwise_180", "clockwise_270"
        ] = "clockwise_0",
        start: int = 0,
        end: Optional[int] = None,
        boundary_grad_quantile: float = 0.95,
        bkgd_kernel_size: int = 60,
        depth_far: float = 3,
        dump_visual: bool = True,
        suppress_bad_frames_validation: bool = False,
    ):
        self.sequence = sequence
        self.frgd_prompts = frgd_prompts
        assert len(frgd_prompts) in [1, 2], "MTTR expects 1 or 2 prompts."
        self.data_root = data_root or osp.join(
            osp.dirname(__file__), "..", "..", "datasets", "record3d"
        )
        self.dump_root = dump_root or osp.abspath(
            osp.join(osp.dirname(__file__), "..", "..", "datasets", "iphone")
        )
        self.rotate_mode = rotate_mode
        self.start = start
        self.end = end
        self.boundary_grad_quantile = boundary_grad_quantile
        self.bkgd_kernel_size = bkgd_kernel_size
        self.depth_far = depth_far
        self.dump_visual = dump_visual
        self.suppress_bad_frames_validation = suppress_bad_frames_validation

        self.factors = (1, DEFAULT_FACTOR)
        self.data_dir = osp.join(self.data_root, sequence)
        self.dump_dir = osp.join(self.dump_root, sequence)
        self.rotate_transfm = utils.rotate_transfm(self.rotate_mode)
        if self.end is None:
            ply_zip_paths = path_ops.ls(
                osp.join(self.data_dir, "Zipped_PLY/*.zip")
            )
            assert len(ply_zip_paths) == 1
            ply_zip_path = ply_zip_paths[0]
            with ZipFile(ply_zip_path) as ply_zip:
                names = sorted(ply_zip.namelist())
            self.end = len(names)

        if self.has_novel_view:
            self.validate_novel_view_data()

        metadata = io.load(osp.join(self.data_dir, "RGBD_Video/metadata.json"))
        self.fps = float(metadata["fps"])

    @property
    def has_novel_view(self):
        return osp.exists(osp.join(self.data_dir, "Test"))

    def validate_novel_view_data(self):
        metadata = io.load(osp.join(self.data_dir, "RGBD_Video/metadata.json"))
        W, H, fps = (
            metadata["w"],
            metadata["h"],
            float(metadata["fps"]),
        )

        val_dirs = path_ops.ls(osp.join(self.data_dir, "Test/*"), type="d")
        for val_dir in val_dirs:
            if not osp.exists(
                osp.join(val_dir, "RGBD_Video/timesync_offset.txt")
            ):
                raise FileNotFoundError(
                    f"Time sync file not found in {val_dir}. If not "
                    f"synchrnoized before, please consider running "
                    f"tools/combine_record3d_av.py and Premiere Pro "
                    f"audio-based multi-camera synchronization "
                    f"(See https://www.youtube.com/watch?v=57Aek730DNU "
                    f"for example)."
                )

            if (
                not osp.exists(
                    osp.join(val_dir, "RGBD_Video/timesync_bad_frames.txt")
                )
                and not self.suppress_bad_frames_validation
            ):
                raise FileNotFoundError(
                    f"Time sync bad frame file not found in {val_dir}. If not "
                    f"annotated before, please consider running "
                    f"tools/annotate_record3d_val_frames.ipynb."
                )

            metadata = io.load(osp.join(val_dir, "RGBD_Video/metadata.json"))
            assert (
                metadata["w"] == W
                and metadata["h"] == H
                and float(metadata["fps"]) == fps
            )

    def _process_train_points(self):
        def process_zipped_ply(obj, name):
            pcd = trimesh.load(
                trimesh.util.wrap_as_stream(obj.read(name)), file_type="ply"
            )
            points = geometry.matv(
                self.rotate_transfm, np.asarray(pcd.vertices)
            )
            # Point RGBs in uint8.
            point_rgbs = np.asarray(pcd.colors)[:, :3]
            return points, point_rgbs

        ply_zip_paths = path_ops.ls(
            osp.join(self.data_dir, "Zipped_PLY/*.zip")
        )
        assert len(ply_zip_paths) == 1
        ply_zip_path = ply_zip_paths[0]
        with ZipFile(ply_zip_path) as ply_zip:
            names = sorted(ply_zip.namelist())
            names = names[self.start : self.end]
            self._train_points, self._train_point_rgbs = common.tree_collate(
                common.parallel_map(
                    lambda name: process_zipped_ply(ply_zip, name),
                    names,
                    show_pbar=True,
                    desc="* Processing train points",
                ),
                collate_fn=lambda *x: x,
            )

    def _process_train_cameras(self):
        def process_cameras(data_dir, use_static):
            metadata_path = osp.join(data_dir, "RGBD_Video/metadata.json")
            metadata = io.load(metadata_path)
            K = np.array(metadata["K"], np.float32).reshape(3, 3).T
            img_wh = (metadata["w"], metadata["h"])
            K, img_wh = utils.rotate_intrin(K, img_wh, self.rotate_mode)
            if not use_static:
                c2ws = np.array(metadata["poses"], np.float32)
                offset_path = osp.join(
                    data_dir, "RGBD_Video/timesync_offset.txt"
                )
                if osp.exists(offset_path):
                    offset = int(np.loadtxt(offset_path))
                    c2ws = c2ws[offset:]
                c2ws = c2ws[self.start : self.end]
                # (T, 3, 4)
                c2ws = np.concatenate(
                    [
                        Rotation.from_quat(c2ws[:, :4]).as_matrix(),
                        c2ws[:, 4:, None],
                    ],
                    axis=-1,
                )
                c2ws = utils.rotate_c2ws(c2ws, self.rotate_mode)
                # Convert OpenGL to OpenCV camera coordinate system.
                c2ws = c2ws @ np.diag([1, -1, -1, 1])
            else:
                c2ws = np.eye(3, 4, dtype=np.float32)[None].repeat(
                    self.end - self.start, axis=0  # type: ignore
                )
            return [
                geometry.Camera(
                    orientation=c2ws[i, :, :3].T,
                    position=c2ws[i, :, -1],
                    focal_length=K[0, 0],
                    principal_point=K[:2, -1],
                    image_size=img_wh,
                )
                for i in range(c2ws.shape[0])
            ]

        logging.info("* Processing train cameras.")
        self.train_cameras = process_cameras(self.data_dir, use_static=False)
        if self.has_novel_view:
            # Identity cameras with unknown relative pose to training cameras.
            # Require further calibration.
            self._val_cameras = [
                process_cameras(d, use_static=True)
                for d in path_ops.ls(
                    osp.join(self.data_dir, "Test/*"), type="d"
                )
            ]

    def _process_train_rgbas_depths(self):
        assert hasattr(self, "_train_points") and hasattr(
            self, "_train_point_rgbs"
        ), (
            "Points must be processed before depth. Consider running "
            "self._process_all_points first."
        )
        assert hasattr(self, "train_cameras"), (
            "Cameras must be processed before depth. Consider running "
            "self._process_all_cameras first."
        )

        def process_rgba_depth(points, point_rgbs, camera):
            pixels, point_depths = camera.project(points, return_depth=True)
            # Convert xy to natural yx indexing.
            pixels = np.round(pixels[..., ::-1]).astype(np.int32)
            mask = (
                (pixels[..., 0] >= 0)
                & (pixels[..., 0] < camera.image_shape[0])
                & (pixels[..., 1] >= 0)
                & (pixels[..., 1] < camera.image_shape[1])
            )
            pixels = pixels[mask]
            points = points[mask]
            point_depths = point_depths[mask]
            point_rgbs = point_rgbs[mask]
            # There are going to be cases where some pixels are missed in
            # Lidar.
            rgb = np.zeros(tuple(camera.image_shape) + (3,), dtype=np.uint8)
            rgb[tuple(pixels.T)] = point_rgbs
            depth = np.zeros(
                tuple(camera.image_shape) + (1,), dtype=np.float32
            )
            depth[tuple(pixels.T)] = point_depths

            alpha = (depth != 0).astype(np.uint8) * 255
            rgba = np.concatenate([rgb, alpha], axis=-1)

            img_points = np.zeros(
                tuple(camera.image_shape) + (3,), dtype=np.float32
            )
            img_points[tuple(pixels.T)] = points
            boundary_mask = utils.sobel_by_quantile(
                img_points, q=self.boundary_grad_quantile
            )
            # NOTE(Hang Gao @ 08/18): Record3D's depth passes pixel corner
            # rather than center. Since the depth is already coarse, we can
            # still use center for the NeRF rendering.
            assert camera.use_projective_depth
            depth = np.minimum(depth, self.depth_far)
            depth = np.where(boundary_mask == 255, 0, depth)

            return rgba, depth, img_points

        logging.info("* Processing train RGBAs and depths.")
        # NOTE(Hang Gao @ 08/18): the RGB video from Record3D is slightly
        # misaligned with the Lidar sensor temporally. We opt to directly use
        # the RGB recording from the point clouds. Note that this will cause
        # some issue for temporal synchronization with validation cameras, but
        # empirically we find that error is very small (less than 10 ms).
        (
            self.train_rgbas,
            self.train_depths,
            self._train_points,
        ) = common.tree_collate(
            common.parallel_map(
                process_rgba_depth,
                self._train_points,
                self._train_point_rgbs,
                self.train_cameras,
                show_pbar=True,
                desc="* Processing train RGBAs and depths",
            ),
        )
        self.train_points = self._train_points[self.train_depths[..., 0] != 0]

    def _process_train_masks(self):
        logging.info("* Processing train masks.")
        compute_mttr_video_mask = processors.get_compute_mttr_video_mask()
        self._train_masks = compute_mttr_video_mask(
            self.train_rgbas[..., :3],
            self.frgd_prompts,
            masks=image.to_float32(self.train_rgbas[..., 3:]),
        )
        self.train_masks = sum(self._train_masks).clip(max=1)

    def _process_scene(self):
        logging.info("* Processing scene.")

        train_points = np.concatenate(
            [
                self.train_points,
                np.array([c.position for c in self.train_cameras], np.float32),
            ],
            axis=0,
        )
        self.bbox = np.stack(
            [train_points.min(axis=0), train_points.max(axis=0)]
        )
        self.center = self.bbox.mean(axis=0)
        self.scale = 1 / np.linalg.norm(self.bbox.ptp(axis=0))
        dists = np.concatenate(
            common.parallel_map(
                lambda p, d, c: np.linalg.norm(
                    p[d[..., 0] != 0] - c.position,
                    axis=-1,
                ),
                self._train_points,
                self.train_depths,
                self.train_cameras,
            )
        )
        self.near = np.quantile(dists, 0.001) * 3 / 4
        self.far = np.quantile(dists, 0.999) * 5 / 4

        bkgd_masks = 1 - np.array(
            common.parallel_map(
                lambda img: utils.dilate(img, self.bkgd_kernel_size),
                self.train_masks,
            )
        )
        self.bkgd_points, self.bkgd_point_rgbs = utils.tsdf_fusion(
            self.train_rgbas[..., :3],
            np.where(bkgd_masks == 1, self.train_depths[..., 0], 0),
            self.train_cameras,
            voxel_length=self.bbox.ptp(axis=0).mean() / 512,
            depth_far=self.depth_far,
        )

        self.lookat = geometry.utils.tringulate_rays(
            np.stack([c.position for c in self.train_cameras], axis=0),
            np.stack([c.optical_axis for c in self.train_cameras], axis=0),
        )
        self.up = np.mean([c.up_axis for c in self.train_cameras], axis=0)

    def _visualize_scene(self):
        logging.info("* Visualizing scene.")
        bbox_points, bbox_end_points = utils.get_bbox_segments(self.bbox)
        plots = {
            "bkgd_points": visuals.PointCloud(
                common.random_subset(
                    self.bkgd_points, min(len(self.bkgd_points), 20000)
                ),
                image.to_uint8(
                    common.random_subset(
                        self.bkgd_point_rgbs, min(len(self.bkgd_points), 20000)
                    )
                ),
            ),
            "bbox": visuals.Segment(
                points=bbox_points,
                end_points=bbox_end_points,
                rgbs=[255, 0, 0],
            ),
            "lookat": {
                "obj": visuals.PointCloud(
                    points=self.lookat[None], rgbs=[255, 0, 0]
                ),
                "marker_size": 10,
            },
        }
        fig = visuals.visualize_scene(plots, height=800, width=800)
        fig.write_html(osp.join(self.dump_dir, "visual", "scene.html"))

    def _visualize_train_video(self):
        logging.info("* Visualizing train video.")

        def visualize_pcd_renderings(camera, depth, rgb):
            points = camera.pixels_to_points(camera.get_pixels(), depth)[
                depth[..., 0] != 0
            ]
            point_rgbs = rgb[depth[..., 0] != 0]
            return visuals.visualize_pcd_renderings(
                points, point_rgbs, self.train_cameras[0]
            ).rgb

        video_visual = np.concatenate(
            [
                self.train_rgbas[..., :3],
                common.parallel_map(
                    lambda d: visuals.visualize_depth(d, invalid_depth=0),
                    self.train_depths,
                    show_pbar=True,
                    desc="* Visualizing depth video",
                ),
                image.to_uint8(self.train_masks.repeat(3, axis=-1)),
                common.parallel_map(
                    visualize_pcd_renderings,
                    self.train_cameras,
                    self.train_depths,
                    self.train_rgbas[..., :3],
                    show_pbar=True,
                    desc="* Visualizing pcd video",
                ),
            ],
            axis=2,
        )
        dump_path = osp.join(self.dump_dir, "visual", "train.mp4")
        io.dump(dump_path, video_visual, fps=self.fps)
        logging.info(f'    Dumped to "{dump_path}".')

    def _process_val_rgbas(self):
        def process_rgbas(data_dir):
            video_paths = path_ops.ls(osp.join(data_dir, "RGBD_Video/*.mp4"))
            assert len(video_paths) == 1
            video_path = video_paths[0]

            offset = 0
            offset_path = osp.join(data_dir, "RGBD_Video/timesync_offset.txt")
            if osp.exists(offset_path):
                offset = int(np.loadtxt(offset_path))

            bad_frames = []
            bad_frames_path = osp.join(
                data_dir, "RGBD_Video/timesync_bad_frames.txt"
            )
            if osp.exists(bad_frames_path):
                # Already offset.
                bad_frames = np.loadtxt(bad_frames_path, dtype=np.int32)
                bad_frames = [
                    i for i in bad_frames if self.start <= i < self.end
                ]

            vid_metadata = io.load_vid_metadata(video_path)
            W = vid_metadata["width"]

            rgbas = io.load(
                video_path,
                trim_kwargs={
                    "start_frame": offset + self.start,
                    "end_frame": offset + self.end,
                },
            )[..., W // 2 :, :]
            rgbas = np.concatenate(
                [rgbas, np.full_like(rgbas[..., :1], 255, dtype=np.uint8)],
                axis=-1,
            )

            return rgbas, bad_frames

        if self.has_novel_view:
            logging.info("* Processing val RGBs.")
            # Might be shorter than the training sequence.
            self.val_rgbas, self.val_bad_frames = list(
                zip(
                    *[
                        process_rgbas(d)
                        for d in path_ops.ls(
                            osp.join(self.data_dir, "Test/*"), type="d"
                        )
                    ],
                )
            )

    def _process_val_cameras(
        self,
        *,
        trees: int = 5,
        checks: int = 50,
        min_match_count: int = 25,
    ):
        if self.has_novel_view:
            train_masks = self.train_rgbas[..., 3:]
            train_grays = np.array(
                [
                    cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
                    for i in self.train_rgbas[..., :3]
                ]
            )
            val_grays = [
                np.array(
                    [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in vi[..., :3]]
                )
                for vi in self.val_rgbas
            ]

            sift = cv2.SIFT_create()
            flann = cv2.FlannBasedMatcher(
                {"algorithm": 0, "trees": trees}, {"checks": checks}
            )

            def detect_sift_kps(gray, mask=None):
                kps, descs = sift.detectAndCompute(gray, None)

                if mask is None:
                    return kps, descs

                pts = np.array([kp.pt for kp in kps], np.float32)
                # remap function starts from (0, 0) at corner.
                kp_masks = (
                    cv2.remap(
                        mask,
                        pts[..., 0],
                        pts[..., 1],
                        interpolation=cv2.INTER_AREA,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    == 1
                )[..., 0]
                kp_inds = np.arange(len(kps))[kp_masks]
                return tuple(kps[i] for i in kp_inds), descs[kp_inds]

            # SIFT detects keypoints starting from (0, 0) at corner.
            train_kps, train_descs = list(
                zip(
                    *common.parallel_map(
                        detect_sift_kps,
                        train_grays,
                        (
                            (train_masks == 255) & (self.train_depths != 0)
                        ).astype(np.float32),
                        show_pbar=True,
                        desc="* Processing train descs",
                    )
                )
            )
            val_kps, val_descs = [], []
            for vg in common.tqdm(
                val_grays,
                desc=f"* Processing val descs",
                position=0,
            ):
                _val_kps, _val_descs = list(
                    zip(
                        *common.parallel_map(
                            detect_sift_kps,
                            vg,
                            show_pbar=True,
                            pbar_kwargs={"position": 1, "leave": False},
                        )
                    )
                )
                val_kps.append(_val_kps)
                val_descs.append(_val_descs)

            def match_points_pixels(
                camera, depth, kps, descs, kps_to, descs_to
            ):
                matches = flann.knnMatch(descs, descs_to, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                if len(good_matches) > min_match_count:
                    pixels = np.array(
                        [kps[m.queryIdx].pt for m in good_matches], np.float32
                    ).reshape(-1, 2)
                    pixels_to = np.array(
                        [kps_to[m.trainIdx].pt for m in good_matches],
                        np.float32,
                    ).reshape(-1, 2)
                else:
                    return (
                        np.empty((0, 3), np.float32),
                        np.empty((0, 2), np.float32),
                        np.empty((0, 2), np.int32),
                    )

                # remap function starts from (0, 0) at corner.
                points = camera.pixels_to_points(
                    pixels,
                    cv2.remap(
                        depth,
                        pixels[..., None, 0],
                        pixels[..., None, 1],
                        interpolation=cv2.INTER_AREA,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    ),
                )
                return points, pixels, pixels_to

            val_cameras, val_matches = [], []
            for _val_kps, _val_descs, _val_cameras in zip(
                common.tqdm(
                    val_kps,
                    desc="* Solving val cameras by PnP",
                    position=0,
                ),
                val_descs,
                self._val_cameras,
            ):
                Tmin = min([len(_val_kps), len(train_kps)])
                points, _pixels, _pixels_to = common.tree_collate(
                    common.parallel_map(
                        match_points_pixels,
                        self.train_cameras[:Tmin],
                        self.train_depths[:Tmin],
                        train_kps[:Tmin],
                        train_descs[:Tmin],
                        _val_kps[:Tmin],
                        _val_descs[:Tmin],
                        show_pbar=True,
                        pbar_kwargs={"position": 1, "leave": False},
                    ),
                    collate_fn=lambda *x: x,
                )
                points = np.concatenate(points, axis=0)
                pixels = np.concatenate(_pixels, axis=0)
                pixels_to = np.concatenate(_pixels_to, axis=0)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    points,
                    pixels_to,
                    _val_cameras[0].intrin,
                    np.zeros(4),
                    flags=cv2.SOLVEPNP_P3P,
                    iterationsCount=1000,
                )
                if not success:
                    logging.error("PnP failed. Somethiing is wrong.")
                    __import__("ipdb").set_trace()

                orientation = cv2.Rodrigues(rvec)[0]
                position = geometry.matv(-orientation.T, tvec[..., 0])
                new_val_cameras, new_val_matches = [], []
                ends = np.cumsum([len(p) for p in _pixels])
                pi = 0
                for ti, c in enumerate(_val_cameras[:Tmin]):
                    c = c.copy()
                    c.orientation = orientation
                    c.position = position
                    qi = pi
                    while pi < len(inliers) and inliers[pi] < ends[ti]:
                        pi += 1
                    new_val_cameras.append(c)
                    cpoints = points[inliers[qi:pi]]
                    cpixels = pixels[inliers[qi:pi]]
                    cpixels_to = c.project(cpoints)
                    new_val_matches.append(
                        np.concatenate([cpixels, cpixels_to], axis=-2)
                    )

                val_cameras.append(new_val_cameras)
                val_matches.append(new_val_matches)

            self.val_cameras = val_cameras
            self.val_matches = val_matches

    def _visualize_val_video(self):
        if self.has_novel_view:
            logging.info("* Visualizing val video.")
            video_visual = [
                np.array(
                    common.parallel_map(
                        lambda corrs, tg, vg: visuals.visualize_corrs(
                            corrs,
                            tg[..., :3],
                            vg[..., :3],
                            rgbs=np.array([0, 255, 0], np.uint8),
                            min_rad=0,
                            subsample=1,
                            num_min_keeps=0,
                            circle_radius=10,
                            circle_thickness=-1,
                        ),
                        self.val_matches[vi],
                        self.train_rgbas[: len(self.val_matches[vi])],
                        self.val_rgbas[vi],
                        show_pbar=True,
                        desc="* Visualizing corrs video",
                        pbar_kwargs={"leave": False},
                    )
                )
                for vi in range(len(self.val_rgbas))
            ]
            Tmax = max([len(v) for v in video_visual])
            video_visual = np.concatenate(
                [
                    np.pad(v, ((0, Tmax - v.shape[0]), (0, 0), (0, 0), (0, 0)))
                    for v in video_visual
                ],
                axis=2,
            )
            dump_path = osp.join(self.dump_dir, "visual", "val.mp4")
            io.dump(dump_path, video_visual, fps=self.fps)
            logging.info(f'    Dumped to "{dump_path}".')

    def _dump_data(self):
        logging.info("* Dumping data.")

        train_ids = [f"0_{i:05d}" for i in range(len(self.train_rgbas))]
        if self.has_novel_view:
            val_frame_masks = [
                i not in bad_frames
                for v, bad_frames in enumerate(self.val_bad_frames)
                for i in range(len(self.val_rgbas[v]))
            ]

            def filter_val_items(items):
                if not isinstance(items, np.ndarray):
                    return [
                        item
                        for i, item in enumerate(items)
                        if val_frame_masks[i]
                    ]
                else:
                    return items[val_frame_masks]

            val_ids = filter_val_items(
                [
                    f"{v+1}_{i:05d}"
                    for v in range(len(self.val_rgbas))
                    for i in range(len(self.val_rgbas[v]))
                ]
            )
        else:
            val_ids = []

        parallel_map = functools.partial(
            common.parallel_map,
            show_pbar=True,
            pbar_kwargs={"position": 1, "leave": False},
        )

        def _rescale_rgba(rgba, factor):
            if factor == 1:
                return rgba
            rgb = image.rescale(
                rgba[..., :3],
                scale_factor=1 / factor,
                interpolation=cv2.INTER_AREA,
            )
            alpha = (
                image.rescale(
                    rgba[..., 3],
                    scale_factor=1 / factor,
                    interpolation=cv2.INTER_AREA,
                )
                == 255
            ).astype(np.uint8) * 255
            return np.concatenate([rgb, alpha[..., None]], axis=-1)

        def _rescale_depth(depth, factor):
            if factor == 1:
                return depth
            mask = image.rescale(
                (depth != 0).astype(np.uint8) * 255,
                scale_factor=1 / factor,
                interpolation=cv2.INTER_AREA,
            )
            depth = image.rescale(
                depth,
                scale_factor=1 / factor,
                interpolation=cv2.INTER_AREA,
            )[..., None]
            depth[mask != 255] = 0
            return depth

        parallel_map(
            io.dump,
            [
                osp.join(
                    self.dump_dir,
                    "camera",
                    train_id + ".json",
                )
                for train_id in train_ids
            ],
            [c.asdict() for c in self.train_cameras],
            desc="* Dumping train cameras",
        )
        if self.has_novel_view:
            parallel_map(
                io.dump,
                [
                    osp.join(
                        self.dump_dir,
                        "camera",
                        val_id + ".json",
                    )
                    for val_id in val_ids
                ],
                filter_val_items(  # type: ignore
                    [c.asdict() for c in sum(self.val_cameras, [])]
                ),
                desc="* Dumping val cameras",
            )

        pbar = common.tqdm(self.factors, position=0)
        for factor in pbar:
            pbar.set_description(f"* Dumping imgs at {factor}x")
            parallel_map(
                lambda p, img: io.dump(p, _rescale_rgba(img, factor)),
                [
                    osp.join(
                        self.dump_dir,
                        "rgb",
                        f"{factor}x",
                        train_id + ".png",
                    )
                    for train_id in train_ids
                ],
                list(self.train_rgbas),
                desc="* Dumping train RGBAs",
            )
            parallel_map(
                lambda p, img: io.dump(p, _rescale_depth(img, factor)),
                [
                    osp.join(
                        self.dump_dir,
                        "depth",
                        f"{factor}x",
                        train_id + ".npy",
                    )
                    for train_id in train_ids
                ],
                self.train_depths,
                desc="* Dumping train depths",
            )
            if self.has_novel_view:
                parallel_map(
                    lambda p, img: io.dump(p, _rescale_rgba(img, factor)),
                    [
                        osp.join(
                            self.dump_dir,
                            "rgb",
                            f"{factor}x",
                            val_id + ".png",
                        )
                        for val_id in val_ids
                    ],
                    filter_val_items(np.concatenate(self.val_rgbas, axis=0)),  # type: ignore
                    desc="* Dumping val RGBAs",
                )

        logging.info("* Dumping scene info.")
        scene_dict = {
            "scale": self.scale,
            "center": self.center,
            "near": self.near * self.scale,
            "far": self.far * self.scale,
        }
        io.dump(osp.join(self.dump_dir, "scene.json"), scene_dict)

        logging.info("* Dumping dataset info.")
        dataset_dict = {
            "count": len(train_ids) + len(val_ids),
            "num_exemplars": len(train_ids),
            "ids": train_ids + val_ids,
            "train_ids": train_ids,
            "val_ids": val_ids,
        }
        io.dump(osp.join(self.dump_dir, "dataset.json"), dataset_dict)

        logging.info("* Dumping metadata info.")
        metadata_dict = {
            **{
                item_id: {
                    "warp_id": int(item_id.split("_")[1]),
                    "appearance_id": int(item_id.split("_")[1]),
                    "camera_id": int(item_id.split("_")[0]),
                }
                for item_id in train_ids + val_ids
            },
            **{},
        }
        io.dump(osp.join(self.dump_dir, "metadata.json"), metadata_dict)

        logging.info("* Dumping extra info.")
        extra_dict = {
            "factor": DEFAULT_FACTOR,
            "fps": self.fps,
            "bbox": (self.bbox - self.center) * self.scale,
            "lookat": (self.lookat - self.center) * self.scale,
            "up": self.up,
        }
        io.dump(osp.join(self.dump_dir, "extra.json"), extra_dict)

        logging.info("* Dumping bkgd points.")
        io.dump(osp.join(self.dump_dir, "points.npy"), self.bkgd_points)

    def process(self):
        self._process_train_points()
        self._process_train_cameras()
        self._process_train_rgbas_depths()
        self._process_train_masks()
        self._visualize_train_video()

        self._process_val_rgbas()
        self._process_val_cameras()
        self._visualize_val_video()

        self._process_scene()
        self._visualize_scene()

        self._dump_data()


@gin.configurable()
class iPhoneParser(NerfiesParser):
    """Parser for the Nerfies dataset."""

    SPLITS = SPLITS

    def __init__(
        self,
        dataset: str,
        sequence: str,
        *,
        data_root: Optional[types.PathType] = None,
    ):
        super(NerfiesParser, self).__init__(
            dataset, sequence, data_root=data_root
        )
        self.use_undistort = False

        (
            self._center,
            self._scale,
            self._near,
            self._far,
        ) = _load_scene_info(self.data_dir)
        (
            self._frame_names_map,
            self._time_ids,
            self._camera_ids,
        ) = _load_metadata_info(self.data_dir)
        self._load_extra_info()

        self.splits_dir = osp.join(self.data_dir, "splits")
        if not osp.exists(self.splits_dir):
            self._create_splits()

    def load_rgba(self, time_id: int, camera_id: int) -> np.ndarray:
        return super().load_rgba(time_id, camera_id, use_undistort=False)

    def load_depth(
        self,
        time_id: int,
        camera_id: int,
    ) -> np.ndarray:
        frame_name = self._frame_names_map[time_id, camera_id]
        depth_path = osp.join(
            self.data_dir, "depth", f"{self._factor}x", frame_name + ".npy"
        )
        depth = io.load(depth_path) * self.scale
        camera = self.load_camera(time_id, camera_id)
        # The original depth data is projective; convert it to ray traveling
        # distance.
        depth = depth / camera.pixels_to_cosa(camera.get_pixels())
        return depth

    def load_camera(
        self, time_id: int, camera_id: int, **_
    ) -> geometry.Camera:
        return super().load_camera(time_id, camera_id, use_undistort=False)

    def load_covisible(
        self, time_id: int, camera_id: int, split: str, **_
    ) -> np.ndarray:
        return super().load_covisible(
            time_id, camera_id, split, use_undistort=False
        )

    def load_keypoints(
        self, time_id: int, camera_id: int, split: str, **_
    ) -> np.ndarray:
        return super().load_keypoints(
            time_id, camera_id, split, use_undistort=False
        )

    def _load_extra_info(self) -> None:
        extra_path = osp.join(self.data_dir, "extra.json")
        extra_dict = io.load(extra_path)
        self._factor = extra_dict["factor"]
        self._fps = extra_dict["fps"]
        self._bbox = np.array(extra_dict["bbox"], dtype=np.float32)
        self._lookat = np.array(extra_dict["lookat"], dtype=np.float32)
        self._up = np.array(extra_dict["up"], dtype=np.float32)

    def _create_splits(self):
        def _create_split(split):
            assert split in self.SPLITS, f'Unknown split "{split}".'

            if split == "train":
                mask = self.camera_ids == 0
            elif split == "val":
                mask = self.camera_ids != 0
            else:
                raise ValueError(f"Unknown split {split}.")

            frame_names = self.frame_names[mask]
            time_ids = self.time_ids[mask]
            camera_ids = self.camera_ids[mask]
            split_dict = {
                "frame_names": frame_names,
                "time_ids": time_ids,
                "camera_ids": camera_ids,
            }
            io.dump(osp.join(self.splits_dir, f"{split}.json"), split_dict)

        common.parallel_map(_create_split, self.SPLITS)


@gin.configurable()
class iPhoneDatasetFromAllFrames(NerfiesDatasetFromAllFrames):

    __parser_cls__: Parser = iPhoneParser

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.training:
            self.depths = np.array(
                common.parallel_map(
                    self.parser.load_depth,
                    self._time_ids,
                    self._camera_ids,
                )
            ).reshape(-1, 1)

    def fetch_data(self, index: int):
        """Fetch the data (it maybe cached for multiple batches)."""
        if not self.training:
            data = super().fetch_data(index)
            time_id, camera_id = self.time_ids[index], self.camera_ids[index]
            try:
                data["depth"] = self.parser.load_depth(time_id, camera_id)
            except FileNotFoundError:
                pass
            return data

        return {
            "rgb": self.rgbs,
            "depth": self.depths,
            "mask": self.masks,
            "rays": self.rays,
        }

    @property
    def has_novel_view(self):
        return (
            len(io.load(osp.join(self.data_dir, "dataset.json"))["val_ids"])
            > 0
        )
