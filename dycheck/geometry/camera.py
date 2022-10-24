#!/usr/bin/env python3
#
# File   : camera.py
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

import copy
from typing import Optional, Tuple, Union

import gin
import jax.numpy as jnp
import numpy as np

from dycheck.utils import io, struct, types

from . import utils


def _compute_residual_and_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Auxiliary function of radial_and_tangential_undistort()."""
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + k3 * r))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + 3.0 * k3 * r)
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes undistorted (x, y) from (xd, yd).

    Note that this function is purely running on CPU and thus could be slow.
    The original Nerfies & HyperNeRF are training on distorted raw images but
    with undistorted rays.
    """
    # Initialize from the distorted point.
    x = xd.copy()
    y = yd.copy()

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = np.where(
            np.abs(denominator) > eps,
            x_numerator / denominator,
            np.zeros_like(denominator),
        )
        step_y = np.where(
            np.abs(denominator) > eps,
            y_numerator / denominator,
            np.zeros_like(denominator),
        )

        x = x + step_x
        y = y + step_y

    return x, y


def points_to_local_points(
    points: np.ndarray,
    extrins: np.ndarray,
) -> np.ndarray:
    """Converts points from world to camera coordinates.

    Args:
        points (np.ndarray): A (..., 3) points tensor in world coordinates.
        extrins (np.ndarray): A (..., 4, 4) camera extrinsic tensor, specifying
            world-to-camera transform.

    Returns:
        np.ndarray: A (..., 3) points tensor in camera coordinates.
    """
    return utils.matv(extrins[..., :3, :3], points) + extrins[..., :3, 3]


def project(
    points: types.Array,
    intrins: types.Array,
    extrins: types.Array,
    radial_distortions: Optional[types.Array] = None,
    tangential_distortions: Optional[types.Array] = None,
    *,
    return_depth: bool = False,
    use_projective_depth: bool = True,
) -> types.Array:
    """Projects 3D points to 2D pixels.

    This function supports batched operation and duck typing between numpy and
    jax.numpy arrays.

    Args:
        points (types.Array): A (..., 3) points tensor.
        intrins (types.Array): A (..., 3, 3) intrinsic matrix tensor.
        extrins (types.Array): A (..., 4, 4) extrinsic matrix tensor.
        radial_distortions (Optional[types.Array]): A (..., 3) radial
            distortion tensor.
        tangential_distortions (Optional[types.Array]): A (..., 2) tangential
            distortion tensor.
        return_depth: Whether to return depth.
        use_projective_depth: Whether to use projective depth.

    Returns:
        np.ndarray: A (..., 2) pixels tensor.
    """
    tensors_to_check = [intrins, extrins]
    if radial_distortions is not None:
        tensors_to_check.append(radial_distortions)
    if tangential_distortions is not None:
        tensors_to_check.append(tangential_distortions)
    if isinstance(points, np.ndarray):
        assert all([isinstance(x, np.ndarray) for x in tensors_to_check])
        np_or_jnp = np
    else:
        assert all([isinstance(x, jnp.ndarray) for x in tensors_to_check])
        np_or_jnp = jnp

    local_points = points_to_local_points(points, extrins)

    normalized_pixels = np_or_jnp.where(
        local_points[..., -1:] != 0,
        local_points[..., :2] / local_points[..., -1:],
        0,
    )
    r2 = (normalized_pixels**2).sum(axis=-1, keepdims=True)

    if radial_distortions is not None:
        # Apply radial distortion.
        radial_scalars = 1 + r2 * (
            radial_distortions[..., 0:1]
            + r2
            * (
                radial_distortions[..., 1:2]
                + r2 * radial_distortions[..., 2:3]
            )
        )
    else:
        radial_scalars = 1

    if tangential_distortions is not None:
        # Apply tangential distortion.
        tangential_deltas = 2 * tangential_distortions * np_or_jnp.prod(
            normalized_pixels,
            axis=-1,
            keepdims=True,
        ) + tangential_distortions[..., ::-1] * (
            r2 + 2 * normalized_pixels**2
        )
    else:
        tangential_deltas = 0

    # Apply distortion.
    normalized_pixels = normalized_pixels * radial_scalars + tangential_deltas

    # Map the distorted ray to the image plane and return the depth.
    pixels = utils.matv(
        intrins,
        np_or_jnp.concatenate(
            [
                normalized_pixels,
                np_or_jnp.ones_like(normalized_pixels[..., :1]),
            ],
            axis=-1,
        ),
    )[..., :2]

    if not return_depth:
        return pixels
    else:
        depths = (
            local_points[..., 2:]
            if use_projective_depth
            else np_or_jnp.linalg.norm(local_points, axis=-1, keepdims=True)
        )
        return pixels, depths


@gin.configurable()
class Camera(object):
    """A generic camera class that potentially distorts rays.

    This camera class uses OpenCV camera model, whhere the local-to-world
    transform assumes (right, down, forward).

    Attributes:
        orientation (np.ndarray): The orientation of the camera of shape (3, 3)
            that maps the world coordinates to local coordinates.
        position (np.ndarray): The position of the camera of shape (3,) in the
            world coordinates.
        focal_length (Union[np.ndarray, float]): The focal length of the camera.
        principal_point (np.ndarray): The principal point of the camera of
            shape (2,)
        image_size (np.ndarray): The image size (W, H).
        skew (Union[np.ndarray, float]): The skewness of the camera.
        pixel_aspect_ratio (Union[np.ndarray, float]): The pixel aspect ratio.
        radial_distortion (Optional[np.ndarray]): The radial distortion of the
            camera of shape (3,).
        tangential_distortion (Optional[np.ndarray]): The tangential distortion
            of the camera of shape (2,).
    """

    def __init__(
        self,
        orientation: np.ndarray,
        position: np.ndarray,
        focal_length: Union[np.ndarray, float],
        principal_point: np.ndarray,
        image_size: np.ndarray,
        skew: Union[np.ndarray, float] = 0.0,
        pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
        radial_distortion: Optional[np.ndarray] = None,
        tangential_distortion: Optional[np.ndarray] = None,
        *,
        use_center: bool = True,
        use_projective_depth: bool = True,
    ):
        """Constructor for camera class."""
        if radial_distortion is None:
            radial_distortion = np.array([0, 0, 0], np.float32)
        if tangential_distortion is None:
            tangential_distortion = np.array([0, 0], np.float32)

        self.orientation = np.array(orientation, np.float32)
        self.position = np.array(position, np.float32)
        self.focal_length = np.array(focal_length, np.float32)
        self.principal_point = np.array(principal_point, np.float32)
        self.image_size = np.array(image_size, np.uint32)

        # Distortion parameters.
        self.skew = np.array(skew, np.float32)
        self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, np.float32)
        self.radial_distortion = np.array(radial_distortion, np.float32)
        self.tangential_distortion = np.array(
            tangential_distortion, np.float32
        )

        self.use_center = use_center
        self.use_projective_depth = use_projective_depth

    @classmethod
    def fromjson(cls, filename: types.PathType):
        camera_dict = io.load(filename)

        # Fix old camera JSON.
        if "tangential" in camera_dict:
            camera_dict["tangential_distortion"] = camera_dict["tangential"]

        return cls(
            orientation=np.asarray(camera_dict["orientation"]),
            position=np.asarray(camera_dict["position"]),
            focal_length=camera_dict["focal_length"],
            principal_point=np.asarray(camera_dict["principal_point"]),
            image_size=np.asarray(camera_dict["image_size"]),
            skew=camera_dict["skew"],
            pixel_aspect_ratio=camera_dict["pixel_aspect_ratio"],
            radial_distortion=np.asarray(camera_dict["radial_distortion"]),
            tangential_distortion=np.asarray(
                camera_dict["tangential_distortion"]
            ),
        )

    def asdict(self):
        return {
            "orientation": self.orientation,
            "position": self.position,
            "focal_length": self.focal_length,
            "principal_point": self.principal_point,
            "image_size": self.image_size,
            "skew": self.skew,
            "pixel_aspect_ratio": self.pixel_aspect_ratio,
            "radial_distortion": self.radial_distortion,
            "tangential_distortion": self.tangential_distortion,
        }

    @property
    def scale_factor_x(self):
        return self.focal_length

    @property
    def scale_factor_y(self):
        return self.focal_length * self.pixel_aspect_ratio

    @property
    def principal_point_x(self):
        return self.principal_point[0]

    @property
    def principal_point_y(self):
        return self.principal_point[1]

    @property
    def has_tangential_distortion(self):
        return any(self.tangential_distortion != 0)

    @property
    def has_radial_distortion(self):
        return any(self.radial_distortion != 0)

    @property
    def distortion(self):
        """Camera distortion parameters compatible with OpenCV.

        Reference:
            https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        """
        return np.concatenate(
            [
                self.radial_distortion[:2],
                self.tangential_distortion,
                self.radial_distortion[-1:],
            ]
        )

    @property
    def image_size_y(self):
        return self.image_size[1]

    @property
    def image_size_x(self):
        return self.image_size[0]

    @property
    def image_shape(self):
        return np.array([self.image_size_y, self.image_size_x], np.uint32)

    @property
    def optical_axis(self):
        return self.orientation[2, :]

    @property
    def up_axis(self):
        return -self.orientation[1, :]

    @property
    def translation(self):
        return -self.orientation @ self.position

    @property
    def intrin(self):
        return np.array(
            [
                [self.scale_factor_x, self.skew, self.principal_point_x],
                [0, self.scale_factor_y, self.principal_point_y],
                [0, 0, 1],
            ],
            np.float32,
        )

    @property
    def extrin(self):
        # 4x4 world-to-camera transform.
        return np.concatenate(
            [
                np.concatenate(
                    [self.orientation, self.translation[..., None]], axis=-1
                ),
                np.array([[0, 0, 0, 1]], np.float32),
            ],
            axis=-2,
        )

    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        y = (pixels[..., 1] - self.principal_point_y) / self.scale_factor_y
        x = (
            pixels[..., 0] - self.principal_point_x - y * self.skew
        ) / self.scale_factor_x

        if self.has_radial_distortion or self.has_tangential_distortion:
            x, y = _radial_and_tangential_undistort(
                x,
                y,
                k1=self.radial_distortion[0],
                k2=self.radial_distortion[1],
                k3=self.radial_distortion[2],
                p1=self.tangential_distortion[0],
                p2=self.tangential_distortion[1],
            )

        y = y * self.scale_factor_y + self.principal_point_y
        x = x * self.scale_factor_x + self.principal_point_x + y * self.skew

        return np.stack([x, y], axis=-1)

    def pixels_to_local_viewdirs(self, pixels: np.ndarray):
        """Return the local ray viewdirs for the provided pixels."""
        y = (pixels[..., 1] - self.principal_point_y) / self.scale_factor_y
        x = (
            pixels[..., 0] - self.principal_point_x - y * self.skew
        ) / self.scale_factor_x

        if self.has_radial_distortion or self.has_tangential_distortion:
            x, y = _radial_and_tangential_undistort(
                x,
                y,
                k1=self.radial_distortion[0],
                k2=self.radial_distortion[1],
                k3=self.radial_distortion[2],
                p1=self.tangential_distortion[0],
                p2=self.tangential_distortion[1],
            )

        viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        return viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)

    def pixels_to_viewdirs(self, pixels: np.ndarray) -> np.ndarray:
        """Return the viewdirs for the provided pixels.

        Args:
            pixels (np.ndarray): (..., 2) tensor or np.array containing 2d
                pixel positions.

        Returns:
            np.ndarray: An array containing the normalized ray directions in
                world coordinates.
        """
        if pixels.shape[-1] != 2:
            raise ValueError("The last dimension of pixels must be 2.")

        batch_shape = pixels.shape[:-1]
        pixels = np.reshape(pixels, (-1, 2))

        local_viewdirs = self.pixels_to_local_viewdirs(pixels)
        viewdirs = utils.matv(self.orientation.T, local_viewdirs)

        # Normalize rays.
        viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = viewdirs.reshape((*batch_shape, 3))
        return viewdirs

    def pixels_to_rays(self, pixels: np.ndarray) -> struct.Rays:
        viewdirs = self.pixels_to_viewdirs(pixels)
        # TODO(Hang Gao @ 07/20): Use viewdirs as directions can be an issue.
        return struct.Rays(
            origins=np.broadcast_to(self.position, viewdirs.shape),
            directions=viewdirs,
        )

    def pixels_to_cosa(self, pixels: np.ndarray) -> np.ndarray:
        rays_through_pixels = self.pixels_to_viewdirs(pixels)
        return (rays_through_pixels @ self.optical_axis)[..., None]

    def pixels_to_points(
        self,
        pixels: np.ndarray,
        depth: np.ndarray,
        use_projective_depth: Optional[bool] = None,
    ) -> np.ndarray:
        """Unproject pixels by their depth.

        Args:
            pixels (np.ndarray): (..., 2) tensor or np.array containing 2d
                pixel positions.
            depth (np.ndarray): (..., 1) tensor or np.array containing the
                depth of the corresponding pixels.
            use_projective_depth (bool): Whether to use the projective depth
                model. If None, use the value of `self.use_projective_depth`.

        Returns:
            np.ndarray: An array containing the 3d points in world coordinates.
        """
        if use_projective_depth is None:
            use_projective_depth = self.use_projective_depth

        rays_through_pixels = self.pixels_to_viewdirs(pixels)
        cosa = 1 if not use_projective_depth else self.pixels_to_cosa(pixels)
        points = rays_through_pixels * depth / cosa + self.position
        return points

    def points_to_local_points(self, points: np.ndarray):
        return points_to_local_points(points, self.extrin)

    def project(
        self,
        points: np.ndarray,
        return_depth: bool = False,
        use_projective_depth: Optional[bool] = None,
    ):
        if use_projective_depth is None:
            use_projective_depth = self.use_projective_depth

        return project(
            points,
            self.intrin,
            self.extrin,
            self.radial_distortion,
            self.tangential_distortion,
            return_depth=return_depth,
            use_projective_depth=use_projective_depth,
        )

    def get_pixels(self, use_center: Optional[bool] = None):
        """Return the pixel at center or corner."""
        if use_center is None:
            use_center = self.use_center
        xx, yy = np.meshgrid(
            np.arange(self.image_size_x, dtype=np.float32),
            np.arange(self.image_size_y, dtype=np.float32),
        )
        offset = 0.5 if use_center else 0
        return np.stack([xx, yy], axis=-1) + offset

    def rescale(self, scale: float) -> "Camera":
        """Rescale the camera."""
        if scale <= 0:
            raise ValueError("scale needs to be positive.")

        camera = self.copy()
        camera.position *= scale
        return camera

    def translate(self, transl: np.ndarray) -> "Camera":
        """Translate the camera."""
        camera = self.copy()
        camera.position += transl
        return camera

    def lookat(
        self,
        position: np.ndarray,
        lookat: np.ndarray,
        up: np.ndarray,
        eps: float = 1e-6,
    ) -> "Camera":
        """Rotate the camera to look at a point.

        Copies the provided vision_sfm camera and returns a new camera that is
        positioned at `camera_position` while looking at `look_at_position`.
        Camera intrinsics are copied by this method. A common value for the
        up_vector is (0, 1, 0).

        Args:
            position (np.ndarray): A (3,) numpy array representing the position
                of the camera.
            lookat (np.ndarray): A (3,) numpy array representing the location
                the camera looks at.
            up (np.ndarray): A (3,) numpy array representing the up direction,
                whose projection is parallel to the y-axis of the image plane.
            eps (float): a small number to prevent divides by zero.

        Returns:
            Camera: A new camera that is copied from the original but is
                positioned and looks at the provided coordinates.

        Raises:
            ValueError: If the camera position and look at position are very
                close to each other or if the up-vector is parallel to the
                requested optical axis.
        """

        look_at_camera = self.copy()
        optical_axis = lookat - position
        norm = np.linalg.norm(optical_axis)
        if norm < eps:
            raise ValueError(
                "The camera center and look at position are too close."
            )
        optical_axis /= norm

        right_vector = np.cross(optical_axis, up)
        norm = np.linalg.norm(right_vector)
        if norm < eps:
            raise ValueError("The up-vector is parallel to the optical axis.")
        right_vector /= norm

        # The three directions here are orthogonal to each other and form a
        # right handed coordinate system.
        camera_rotation = np.identity(3)
        camera_rotation[0, :] = right_vector
        camera_rotation[1, :] = np.cross(optical_axis, right_vector)
        camera_rotation[2, :] = optical_axis

        look_at_camera.position = position
        look_at_camera.orientation = camera_rotation
        return look_at_camera

    def undistort_image_domain(self) -> "Camera":
        """Undistort the image domain of the camera.

        Note that this function only disable the distortion parameters. The
        acutal image undistortion should be taken care of explicitly outside.
        """
        camera = self.copy()
        camera.skew = 0
        camera.radial_distortion = np.zeros(3, dtype=np.float32)
        camera.tangential_distortion = np.zeros(2, dtype=np.float32)
        return camera

    def rescale_image_domain(self, scale: float) -> "Camera":
        """Rescale the image domain of the camera."""
        if scale <= 0:
            raise ValueError("scale needs to be positive.")

        camera = self.copy()
        camera.focal_length *= scale
        camera.principal_point *= scale
        camera.image_size = np.array(
            (
                int(round(self.image_size[0] * scale)),
                int(round(self.image_size[1] * scale)),
            )
        )
        return camera

    def crop_image_domain(
        self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0
    ) -> "Camera":
        """Crop the image domain of the camera.

        The crop parameters may not cause the camera image domain dimensions to
        become non-positive.

        Args:
            left (int): number of pixels by which to reduce (or augment, if
                negative) the image domain at the associated boundary.
            right (int): likewise.
            top (int): likewise.
            bottom (int): likewise.

        Returns:
            Camera: A camera with adjusted image dimensions. The focal length
                is unchanged, and the principal point is updated to preserve
                the original principal axis.
        """

        crop_left_top = np.array([left, top])
        crop_right_bottom = np.array([right, bottom])
        new_resolution = self.image_size - crop_left_top - crop_right_bottom
        new_principal_point = self.principal_point - crop_left_top
        if np.any(new_resolution <= 0):
            raise ValueError(
                "Crop would result in non-positive image dimensions."
            )

        new_camera = self.copy()
        new_camera.image_size = np.array(
            [int(new_resolution[0]), int(new_resolution[1])]
        )
        new_camera.principal_point = np.array(
            [new_principal_point[0], new_principal_point[1]]
        )
        return new_camera

    def copy(self) -> "Camera":
        return copy.deepcopy(self)
