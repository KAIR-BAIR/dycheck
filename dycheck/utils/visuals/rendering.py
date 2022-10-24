#!/usr/bin/env python3
#
# File   : rendering.py
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

import matplotlib

matplotlib.use("agg")

from typing import NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection

from dycheck import geometry

from .. import image


class Renderings(NamedTuple):
    # (H, W, 3) in uint8.
    rgb: Optional[np.ndarray] = None
    # (H, W, 1) in float32.
    depth: Optional[np.ndarray] = None
    # (H, W, 1) in [0, 1].
    acc: Optional[np.ndarray] = None


def visualize_pcd_renderings(
    points: np.ndarray, point_rgbs: np.ndarray, camera: geometry.Camera, **_
) -> Renderings:
    """Visualize a point cloud as a set renderings.

    Args:
        points (np.ndarray): (N, 3) array of points.
        point_rgbs (np.ndarray): (N, 3) array of point colors in either uint8
            or float32.
        camera (geometry.Camera): a camera object containing view information.

    Returns:
        Renderings: the image output object.
    """

    point_rgbs = image.to_uint8(point_rgbs)  # type: ignore

    # Setup the camera.
    W, H = camera.image_size

    # project the 3D points to 2D on image plane
    pixels, depths = camera.project(
        points, return_depth=True, use_projective_depth=True
    )
    pixels = pixels.astype(np.int32)
    mask = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < W)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < H)
        & (depths[:, 0] > 0)
    )

    pixels = pixels[mask]
    rgbs = point_rgbs[mask]
    depths = depths[mask]

    sorted_inds = np.argsort(depths[..., 0])[::-1]
    pixels = pixels[sorted_inds]
    rgbs = rgbs[sorted_inds]
    depths = depths[sorted_inds]

    rgb = np.full((H, W, 3), 255, dtype=np.uint8)
    rgb[pixels[:, 1], pixels[:, 0]] = rgbs

    depth = np.zeros((H, W, 1), dtype=np.float32)
    depth[pixels[:, 1], pixels[:, 0]] = depths

    acc = np.zeros((H, W, 1), dtype=np.float32)
    acc[pixels[:, 1], pixels[:, 0]] = 1

    return Renderings(rgb=rgb, depth=depth, acc=acc)


def _is_front(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Triangle is front facing if its projection on XY plane is clock-wise.
    Z = (
        (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 1, 1] + T[:, 0, 1])
        + (T[:, 2, 0] - T[:, 1, 0]) * (T[:, 2, 1] + T[:, 1, 1])
        + (T[:, 0, 0] - T[:, 2, 0]) * (T[:, 0, 1] + T[:, 2, 1])
    )
    return Z >= 0


def grid_faces(
    h: int, w: int, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Creates mesh face indices from a given pixel grid size.

    Args:
        h (int): image height.
        w (int): image width.
        mask (Optional[np.ndarray], optional): mask of valid pixels. Defaults
            to None.

    Returns:
        faces (np.ndarray): array of face indices. Note that the face indices
            include invalid pixels (they are not excluded).
    """
    if mask is None:
        mask = np.ones((h, w), bool)

    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    faces_1 = np.stack([tl, bl, tr], axis=-1)[
        mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:]
    ]
    faces_2 = np.stack([br, tr, bl], axis=-1)[
        mask[1:, 1:] & mask[:-1, 1:] & mask[1:, :-1]
    ]
    faces = np.concatenate([faces_1, faces_2], axis=0)
    return faces


def visualize_mesh_renderings(
    points: np.ndarray,
    faces: np.ndarray,
    point_rgbs: np.ndarray,
    camera: geometry.Camera,
    **_
):
    """Visualize a mesh as a set renderings.

    Note that front facing triangles are defined in clock-wise orientation.

    Args:
        points (np.ndarray): (N, 3) array of points.
        faces (np.ndarray): (F, 3) array of faces.
        point_rgbs (np.ndarray): (N, 3) array of point colors in either uint8
            or float32.
        camera (geometry.Camera): a camera object containing view information.

    Returns:
        Renderings: the image output object.
    """

    # High quality output.
    DPI = 10.0

    face_rgbs = image.to_float32(point_rgbs[faces]).mean(axis=-2)

    # Setup the camera.
    W, H = camera.image_size

    # Project the 3D points to 2D on image plane.
    pixels, depth = camera.project(
        points, return_depth=True, use_projective_depth=True
    )

    T = pixels[faces]
    Z = -depth[faces][..., 0].mean(axis=1)
    front = _is_front(T)
    T, Z = T[front], Z[front]
    face_rgbs = face_rgbs[front]

    # Sort triangles according to z buffer.
    triangles = T[:, :, :2]
    sorted_inds = np.argsort(Z)
    triangles = triangles[sorted_inds]
    face_rgbs = face_rgbs[sorted_inds]

    # Painter's algorithm using matplotlib.
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], xlim=[0, W], ylim=[H, 0], aspect=1)
    ax.axis("off")

    collection = PolyCollection([], closed=True)
    collection.set_verts(triangles)
    collection.set_linewidths(0.0)
    collection.set_facecolors(face_rgbs)
    ax.add_collection(collection)

    canvas.draw()
    s, _ = canvas.print_to_buffer()
    img = np.frombuffer(s, np.uint8).reshape((H, W, 4))
    plt.close(fig)

    rgb = img[..., :3]
    acc = (img[..., -1:] > 0).astype(np.float32)

    # Depth is not fully supported by maptlotlib yet and we render it as if
    # it's an image by a hack.
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], xlim=[0, W], ylim=[H, 0], aspect=1)
    ax.axis("off")

    collection = PolyCollection([], closed=True)
    collection.set_verts(triangles)
    collection.set_linewidths(0.0)
    Z = -Z[sorted_inds]
    Zmin = Z.min()
    Zmax = Z.max()
    Z = (Z - Zmin) / (Zmax - Zmin)
    collection.set_facecolors(Z[..., None].repeat(3, axis=-1))
    ax.add_collection(collection)

    canvas.draw()
    s, _ = canvas.print_to_buffer()
    depth = (
        image.to_float32(
            np.frombuffer(s, np.uint8).reshape((H, W, 4))[..., :1]
        )
        * (Zmax - Zmin)
        + Zmin
    )
    depth[acc[..., 0] == 0] = 0
    plt.close(fig)

    return Renderings(rgb=rgb, depth=depth, acc=acc)
