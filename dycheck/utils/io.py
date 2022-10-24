#!/usr/bin/env python3
#
# File   : io.py
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

import json
import os.path as osp
import pickle as pkl
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import ffmpeg
import numpy as np

from . import common, image, path_ops, types

_LOAD_REGISTRY, _DUMP_REGISTRY = {}, {}

IMG_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)
VID_EXTS = (".mov", ".avi", ".mpg", ".mpeg", ".mp4", ".mkv", ".wmv", ".gif")


def _register(
    *,
    ext: Optional[Union[str, Sequence[str]]] = None,
) -> Callable:
    if isinstance(ext, str):
        ext = [ext]

    def _inner_register(func: Callable) -> Callable:
        name = func.__name__
        assert name.startswith("load_") or name.startswith("dump_")

        nonlocal ext
        if ext is None:
            ext = ["." + name[5:]]
        for e in ext:
            if name.startswith("load_"):
                _LOAD_REGISTRY[e] = func
            else:
                _DUMP_REGISTRY[e] = func

        return func

    return _inner_register


def _dispatch(
    registry: Dict[str, Callable], name: Literal["load", "dump"]
) -> Callable:
    def _dispatched(filename: types.PathType, *args, **kwargs):
        ext = path_ops.get_ext(filename, match_first=False)
        func = registry[ext]
        if (
            name == "dump"
            and osp.dirname(filename) != ""
            and not osp.exists(osp.dirname(filename))
        ):
            path_ops.mkdir(osp.dirname(filename))
        return func(filename, *args, **kwargs)

    _dispatched.__name__ = name
    return _dispatched


load = _dispatch(_LOAD_REGISTRY, "load")
dump = _dispatch(_DUMP_REGISTRY, "dump")


@_register()
def load_txt(
    filename: types.PathType, *, strip: bool = True, **kwargs
) -> List[str]:
    with open(filename) as f:
        lines = f.readlines(**kwargs)
    if strip:
        lines = [line.strip() for line in lines]
    return lines


@_register()
def dump_txt(filename: types.PathType, obj: List[Any], **_) -> None:
    # Prefer visual appearance over compactness.
    obj = "\n".join([str(item) for item in obj])
    with open(filename, "w") as f:
        f.write(obj)


@_register()
def load_json(filename: types.PathType, **kwargs) -> Dict:
    with open(filename) as f:
        return json.load(f, **kwargs)


@_register()
def dump_json(
    filename: types.PathType,
    obj: Dict,
    *,
    sort_keys: bool = True,
    indent: Optional[int] = 4,
    separators: Tuple[str, str] = (",", ": "),
    **kwargs,
) -> None:
    # Process potential numpy arrays.
    if isinstance(obj, dict):
        obj = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        pass
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist()
    else:
        raise ValueError(f"{type(obj)} is not a supported type.")
    # Prefer visual appearance over compactness.
    with open(filename, "w") as f:
        json.dump(
            obj,
            f,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
            **kwargs,
        )


@_register()
def load_pkl(filename: types.PathType, **kwargs) -> Dict:
    with open(filename, "rb") as f:
        try:
            return pkl.load(f, **kwargs)
        except UnicodeDecodeError as e:
            if "encoding" in kwargs:
                raise e
    return load_pkl(filename, encoding="latin1", **kwargs)


@_register()
def dump_pkl(filename: types.PathType, obj: Dict, **kwargs) -> None:
    with open(filename, "wb") as f:
        pkl.dump(obj, f, **kwargs)


@_register()
def load_npy(
    filename: types.PathType, *, allow_pickle: bool = True, **kwargs
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    return np.load(filename, allow_pickle=allow_pickle, **kwargs)


@_register()
def dump_npy(filename: types.PathType, obj: np.ndarray, **kwargs) -> None:
    np.save(filename, obj, **kwargs)


@_register()
def load_npz(filename: types.PathType, **kwargs) -> np.lib.npyio.NpzFile:
    return np.load(filename, **kwargs)


@_register()
def dump_npz(filename: types.PathType, **kwargs) -> None:
    # Disable positional argument for np.savez.
    np.savez(filename, **kwargs)


@_register(ext=IMG_EXTS)
def load_img(
    filename: types.PathType, *, use_rgb: bool = True, **kwargs
) -> np.ndarray:
    img = cv2.imread(filename, **kwargs)
    if use_rgb and img.shape[-1] >= 3:
        # Take care of RGBA case when flipping.
        img = np.concatenate([img[..., 2::-1], img[..., 3:]], axis=-1)
    return img


@_register(ext=IMG_EXTS)
def dump_img(
    filename: types.PathType,
    obj: np.ndarray,
    *,
    use_rgb: bool = True,
    **kwargs,
) -> None:
    if use_rgb and obj.shape[-1] >= 3:
        obj = np.concatenate([obj[..., 2::-1], obj[..., 3:]], axis=-1)
    cv2.imwrite(filename, image.to_uint8(obj), **kwargs)


def load_vid_metadata(filename: types.PathType) -> np.ndarray:
    assert osp.exists(filename), f"{filename} does not exist!"
    try:
        probe = ffmpeg.probe(filename)
    except ffmpeg.Error as e:
        print("stdout:", e.stdout.decode("utf8"))
        print("stderr:", e.stderr.decode("utf8"))
        raise e
    metadata = next(
        stream
        for stream in probe["streams"]
        if stream["codec_type"] == "video"
    )
    metadata["fps"] = float(eval(metadata["r_frame_rate"]))
    return metadata


@_register(ext=VID_EXTS)
def load_vid(
    filename: types.PathType,
    *,
    quiet: bool = True,
    trim_kwargs: Dict[str, Any] = {},
    **_,
) -> Dict:
    vid_metadata = load_vid_metadata(filename)
    W = int(vid_metadata["width"])
    H = int(vid_metadata["height"])

    stream = ffmpeg.input(filename)
    if len(trim_kwargs) > 0:
        stream = ffmpeg.trim(stream, **trim_kwargs).setpts("PTS-STARTPTS")
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24")
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=quiet)
    out = np.frombuffer(out, np.uint8).reshape([-1, H, W, 3])
    return out.copy()


@_register(ext=VID_EXTS)
def dump_vid(
    filename: types.PathType,
    obj: Union[List[np.ndarray], np.ndarray],
    *,
    fps: float,
    quiet: bool = True,
    show_pbar: bool = True,
    desc: Optional[str] = "* Dumping video",
    **kwargs,
) -> None:
    if not isinstance(obj, np.ndarray):
        obj = np.asarray(obj)
    obj = image.to_uint8(obj)

    H, W = obj.shape[1:3]
    stream = ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgb24",
        s="{}x{}".format(W, H),
        r=fps,
    )
    process = (
        stream.output(filename, pix_fmt="yuv420p", vcodec="libx264")
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=quiet)
    )
    obj_bytes = common.parallel_map(
        lambda f: f.tobytes(),
        list(obj),
        show_pbar=show_pbar,
        desc=desc,
        **kwargs,
    )
    for b in obj_bytes:
        process.stdin.write(b)
    process.stdin.close()
    process.wait()
