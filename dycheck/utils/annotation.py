#!/usr/bin/env python3
#
# File   : annotation.py
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
from typing import List

import cv2
import jax
import numpy as np

from ipyevents import Event
from IPython.display import clear_output, display
from ipywidgets import HTML, Button, HBox, Image, Output

from . import common, visuals


def annotate_record3d_bad_frames(
    frames: np.ndarray,
    *,
    frame_ext: str = ".png",
) -> List[np.ndarray]:
    """Interactively annotate bad frames in validation set to skip later.

    "Bad images" exist because the hand may occlude the scene during capturing.

    Args:
        frames (np.ndarray]): images of shape (N, H, W, 3) in uint8 to annotate.
        skel (Skeleton): a skeleton definition object.
        frame_ext (str): the extension of images. This is used for decoding and
            then display. Default: '.png'.

    Return:
        np.ndarray: bad frame indices of shape (N_bad,).
    """

    def _frame_to_widget(frame):
        value = cv2.imencode(frame_ext, frame[..., ::-1])[1].tobytes()
        widget = Image(value=value, format=frame_ext[1:])
        widget.layout.max_width = "100%"
        widget.layout.height = "auto"
        return widget

    out = Output()

    frame_widgets = list(
        map(
            _frame_to_widget,
            common.tqdm(frames, desc="* Decoding frames"),
        )
    )
    bad_frames = []

    frame_idx = -1
    frame_msg = HTML()

    def show_next_frame():
        nonlocal frame_idx, frame_msg
        frame_idx += 1

        frame_msg.value = (
            f"{frame_idx} frames annotated, "
            f"{len(frames) - frame_idx} frames left. "
        )

        if frame_idx == len(frames):
            for btn in buttons:
                if btn is not redo_btn:
                    btn.disabled = True
            print("Annotation done.")
            return

        frame_widget = frame_widgets[frame_idx]
        with out:
            clear_output(wait=True)
            display(frame_widget)

    def mark_frame(_, good):
        nonlocal frame_idx, bad_frames
        if not good:
            bad_frames.append(frame_idx)

        show_next_frame()

    def redo_frame(btn):
        nonlocal frame_idx, bad_frames
        if len(bad_frames) > 0 and bad_frames[-1] == frame_idx - 1:
            _ = bad_frames.pop()
        frame_idx = max(-1, frame_idx - 2)

        for btn in buttons:
            if btn is not redo_btn:
                btn.disabled = False

        show_next_frame()

    buttons = []

    valid_btn = Button(description="👍")
    valid_btn.on_click(functools.partial(mark_frame, good=True))
    buttons.append(valid_btn)

    invalid_btn = Button(description="👎")
    invalid_btn.on_click(functools.partial(mark_frame, good=False))
    buttons.append(invalid_btn)

    redo_btn = Button(description="♻️")
    redo_btn.on_click(redo_frame)
    buttons.append(redo_btn)

    display(HBox([frame_msg]))
    display(HBox(buttons))
    display(HBox([out]))

    show_next_frame()

    return bad_frames


def annotate_keypoints(
    frames: np.ndarray,
    skeleton: visuals.Skeleton,
    *,
    frame_ext: str = ".png",
    **kwargs,
) -> List[np.ndarray]:
    """Interactively annotate keypoints on input frames.

    Note that each frame will only be submitted when finished.

    Args:
        frames (np.ndarray]): images of shape (N, H, W, 3) in uint8 to annotate.
        skel (Skeleton): a skeleton definition object.
        frame_ext (str): the extension of images. This is used for decoding and
            then display. Default: '.png'.

    Return:
        keypoints (List[np.ndarray]): a list of N annotated keypoints of shape
            (J, 3) where the last column is visibility in [0, 1].
    """

    def _frame_to_widget(frame):
        value = cv2.imencode(frame_ext, np.array(frame[..., ::-1]))[
            1
        ].tobytes()
        widget = Image(value=value, format=frame_ext[1:])
        widget.layout.max_width = "100%"
        widget.layout.height = "auto"
        return widget

    frame_widgets = jax.tree_map(_frame_to_widget, list(frames))
    keypoints = []

    frame_idx, kps = -1, []

    event, frame_msg, kp_msg, kp_inst = Event(), HTML(), HTML(), HTML()

    def show_next_frame():
        nonlocal frame_idx, frame_msg, event
        frame_idx += 1

        frame_msg.value = (
            f"{len(keypoints)} frames annotated, "
            f"{len(frames) - frame_idx} frames left."
        )

        if frame_idx == len(frames):
            _show_current_kp_name()
            for btn in buttons:
                if btn is not redo_btn:
                    btn.disabled = True
            event.watched_events = []
            print("Annotation done.")
            return
        else:
            _show_current_kp_name()
            for btn in buttons:
                btn.disabled = False

        _mark_next_kp()

        frame_widget = frame_widgets[frame_idx]
        with out:
            clear_output(wait=True)
            display(frame_widget)
        event.source = frame_widget
        event.watched_events = ["click"]
        event.on_dom_event(mark_kp)

    def _show_kp_visual():
        padded_kps = np.array(
            kps + [[0, 0, 0] for _ in range(skeleton.num_kps - len(kps))],
            dtype=np.float32,
        )
        canvas = frames[frame_idx]
        kp_visual = visuals.visualize_kps(
            padded_kps, canvas, skeleton=skeleton, **kwargs
        )
        visual_widget = _frame_to_widget(kp_visual)
        with kp_visual_out:
            clear_output(wait=True)
            display(visual_widget)

    def _mark_next_kp():
        _show_kp_visual()
        if len(kps) == skeleton.num_kps:
            submit_frame()

        _show_current_kp_name()

        nonlocal kp_msg
        kp_msg.value = (
            f"{len(kps)} keypoints annotated, "
            f"{len(skeleton.kp_names) - len(kps)} keypoints left."
        )

    def _show_current_kp_name():
        nonlocal frame_idx
        if frame_idx == len(frames):
            msg = "FINISHED!"
        else:
            kp_name = skeleton.kp_names[len(kps)]
            msg = f"Marking <b>[{kp_name}]</b>..."
        kp_inst.value = msg

    def mark_kp(event):
        kp = np.array([event["dataX"], event["dataY"], 1], dtype=np.float32)
        nonlocal kps
        kps.append(kp)

        _mark_next_kp()

    def redo_kp(_):
        nonlocal kps, keypoints, frame_idx
        if len(kps) == 0 and len(keypoints) > 0:
            kps = keypoints.pop().tolist()
            kps = kps[:-1]
            frame_idx -= 2
            show_next_frame()
        else:
            kps.pop()
            _mark_next_kp()

    def mark_invisible_kp(_):
        kp = np.array([0, 0, 0], dtype=np.float32)
        nonlocal kps
        kps.append(kp)

        _mark_next_kp()

    def submit_frame():
        nonlocal kps
        keypoints.append(np.array(kps))
        kps = []

        show_next_frame()

    buttons = []

    invisible_btn = Button(description="🫥")
    invisible_btn.on_click(mark_invisible_kp)
    buttons.append(invisible_btn)

    redo_btn = Button(description="♻️")
    redo_btn.on_click(redo_kp)
    buttons.append(redo_btn)

    display(HBox([kp_inst]))
    display(HBox([frame_msg, kp_msg]))
    display(HBox(buttons))
    out, kp_visual_out = Output(), Output()
    display(HBox([out, kp_visual_out]))
    show_next_frame()

    return keypoints
