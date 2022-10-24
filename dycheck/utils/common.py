#!/usr/bin/env python3
#
# File   : common.py
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
import inspect
from concurrent import futures
from copy import copy
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import jax
import numpy as np


def tolerant_partial(fn: Callable, *args, **kwargs) -> Callable:
    """A thin wrapper around functools.partial which only binds the keyword
    arguments that matches the function signature.
    """
    signatures = inspect.signature(fn)
    return functools.partial(
        fn,
        *args,
        **{k: v for k, v in kwargs.items() if k in signatures.parameters},
    )


def traverse_filter(
    data_dict: Dict[str, Any],
    exclude_fields: Sequence[str] = (),
    return_fields: Sequence[str] = (),
    protect_fields: Sequence[str] = (),
    inplace: bool = False,
) -> Dict[str, Any]:
    """Keep matched field values within the dictionary, either inplace or not.

    Args:
        data_dict (Dict[str, Any]): A dictionary to be filtered.
        exclude_fields (Sequence[str]): A list of fields to be excluded.
        return_fields (Sequence[str]): A list of fields to be returned.
        protect_fields (Sequence[str]): A list of fields to be protected.
        inplace (bool): Whether to modify the input dictionary inplace.

    Returns:
        Dict[str, Any]: The filtered dictionary.
    """
    assert isinstance(data_dict, dict)

    str_to_tupid = lambda s: tuple(s.split("/"))
    exclude_fields = [str_to_tupid(f) for f in set(exclude_fields)]
    return_fields = [str_to_tupid(f) for f in set(return_fields)]
    protect_fields = [str_to_tupid(f) for f in set(protect_fields)]

    filter_fn = lambda f: f in protect_fields or (
        f in return_fields
        if len(return_fields) > 0
        else f not in exclude_fields
    )

    if not inplace:
        data_dict = copy(data_dict)

    def delete_filtered(d, prefix):
        if isinstance(d, dict):
            for k in list(d.keys()):
                path = prefix + (k,)
                if (
                    not isinstance(d[k], dict) or len(d[k]) == 0
                ) and not filter_fn(path):
                    del d[k]
                else:
                    delete_filtered(d[k], path)

    delete_filtered(data_dict, ())
    return data_dict


@functools.lru_cache(maxsize=None)
def in_notebook() -> bool:
    """Check if the code is running in a notebook."""
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if not ipython or "IPKernelApp" not in ipython.config:
            return False
    except ImportError:
        return False
    return True


def tqdm(iterable: Iterable, *args, **kwargs) -> Iterable:
    if not in_notebook():
        from tqdm import tqdm as _tqdm
    else:
        from tqdm.notebook import tqdm as _tqdm
    return _tqdm(iterable, *args, **kwargs)


def parallel_map(
    func: Callable,
    *iterables: Sequence[Iterable],
    max_threads: Optional[int] = None,
    show_pbar: bool = False,
    desc: Optional[str] = None,
    pbar_kwargs: Dict[str, Any] = {},
    debug: bool = False,
    **kwargs,
) -> Sequence[Any]:
    """Parallel version of map()."""
    if not debug:
        with futures.ThreadPoolExecutor(max_threads) as executor:
            if show_pbar:
                results = list(
                    tqdm(
                        executor.map(func, *iterables, **kwargs),
                        desc=desc,
                        total=len(iterables[0]),
                        **pbar_kwargs,
                    )
                )
            else:
                results = list(executor.map(func, *iterables, **kwargs))
            return results
    else:
        return list(map(func, *iterables, **kwargs))


def tree_collate(trees: Sequence[Any], collate_fn=lambda *x: np.asarray(x)):
    """Collates a list of pytrees with the same structure."""
    return jax.tree_multimap(collate_fn, *trees)


def strided_subset(sequence: Sequence[Any], count: int) -> Sequence[Any]:
    if count > len(sequence):
        raise ValueError("count must be less than or equal to len(sequence)")
    inds = np.linspace(0, len(sequence), count, dtype=int, endpoint=False)
    if isinstance(sequence, np.ndarray):
        sequence = sequence[inds]
    else:
        sequence = [sequence[i] for i in inds]
    return sequence


def random_subset(
    sequence: Sequence[Any], count: int, seed: int = 0
) -> Sequence[Any]:
    if count > len(sequence):
        raise ValueError("count must be less than or equal to len(sequence)")
    rng = np.random.default_rng(seed)
    inds = rng.choice(len(sequence), count, replace=False)
    if isinstance(sequence, np.ndarray):
        sequence = sequence[inds]
    else:
        sequence = [sequence[i] for i in inds]
    return sequence
