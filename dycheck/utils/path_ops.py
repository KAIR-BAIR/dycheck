#!/usr/bin/env python3
#
# File   : path_ops.py
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

import glob
import os
import os.path as osp
import re
import shutil
from typing import List

from . import types


def get_ext(
    filename: types.PathType, match_first: bool = False
) -> types.PathType:
    if match_first:
        filename = osp.split(filename)[1]
        return filename[filename.find(".") :]
    else:
        return osp.splitext(filename)[1]


def basename(
    filename: types.PathType, with_ext: bool = True, **kwargs
) -> types.PathType:
    name = osp.basename(filename, **kwargs)
    if not with_ext:
        name = name.replace(get_ext(name), "")
    return name


def natural_sorted(lst: List[types.PathType]) -> List[types.PathType]:
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(lst, key=alphanum_key)


def mtime_sorted(lst: List[types.PathType]) -> List[types.PathType]:
    # Ascending order: last modified file will be the last one.
    return sorted(lst, key=lambda p: os.stat(p).st_mtime)


def ls(
    pattern: str,
    *,
    type: str = "a",
    latestk: int = -1,
    exclude: bool = False,
) -> List[types.PathType]:
    filter_fn = {
        "f": lambda p: osp.isfile(p) and not osp.islink(p),
        "d": osp.isdir,
        "l": osp.islink,
        "a": lambda p: osp.isfile(p) or osp.isdir(p) or osp.islink(p),
    }[type]

    def _natural_sorted_latestk(fs):
        if latestk > 0:
            if not exclude:
                fs = sorted(fs, key=osp.getmtime)[::-1][:latestk]
            else:
                fs = sorted(fs, key=osp.getmtime)[::-1][latestk:]
        return natural_sorted(fs)

    if "**" in pattern:
        dsts = glob.glob(pattern, recursive=True)
    elif "*" in pattern:
        dsts = glob.glob(pattern)
    else:
        dsts = [
            osp.join(pattern, p)
            for p in os.listdir(pattern)
            if filter_fn(osp.join(pattern, p))
        ]
        return _natural_sorted_latestk(dsts)

    dsts = [dst for dst in dsts if filter_fn(dst)]
    return _natural_sorted_latestk(dsts)


def mv(src: types.PathType, dst: types.PathType) -> None:
    shutil.move(src, dst)


def ln(
    src: types.PathType,
    dst: types.PathType,
    use_relpath: bool = True,
    exist_ok: bool = True,
) -> None:
    if osp.exists(dst):
        if exist_ok:
            rm(dst)
        else:
            raise FileExistsError(
                f'Force link from "{src}" to existed "{dst}".'
            )
    if use_relpath:
        src = osp.relpath(src, start=osp.dirname(dst))
    if not osp.exists(osp.dirname(dst)):
        mkdir(osp.dirname(dst))
    os.symlink(src, dst)


def cp(src: types.PathType, dst: types.PathType, **kwargs) -> None:
    try:
        shutil.copyfile(src, dst)
    except OSError:
        shutil.copytree(src, dst, **kwargs)


def mkdir(dst: types.PathType, exist_ok: bool = True, **kwargs) -> None:
    os.makedirs(dst, exist_ok=exist_ok, **kwargs)


def rm(dst: types.PathType) -> None:
    if osp.exists(dst):
        if osp.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        if osp.isfile(dst) or osp.islink(dst):
            os.remove(dst)
