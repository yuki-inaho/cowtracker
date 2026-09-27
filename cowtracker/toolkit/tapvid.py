"""TAP-Vid benchmark pickles as evaluation samples.

Supported containers (the released TAP-Vid / RoboTAP pickles): a dict ``{video_name: entry}`` (DAVIS) or a list of
entries (RGB-Stacking, Kinetics, RoboTAP). Each entry has ``video`` ([T, H, W, 3] uint8, or a list of JPEG bytes),
``points`` ([N, T, 2] normalised (x, y)) and ``occluded`` ([N, T] bool).

The preparation follows AllTracker's ``test_dense_on_sparse.py`` readers: frames are resized to the inference size,
points are scaled by (W, H), tracks visible in fewer than two frames are dropped, and each track is queried at its
first visible frame ("first" query mode).
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from cowtracker.toolkit.video_io import resize_frames

REQUIRED_KEYS = ("video", "points", "occluded")


@dataclass(frozen=True)
class TapVidSample:
    """One benchmark video: frames [T, H, W, 3] uint8, GT points [N, T, 2] px (x, y), occluded [N, T] and
    first-mode queries [N, 3] as (t, y, x)."""

    name: str
    video: np.ndarray
    points: np.ndarray
    occluded: np.ndarray
    query_points: np.ndarray


def read_tapvid_pickle(path: str | Path) -> list[tuple[str, dict]]:
    """(name, entry) pairs of a TAP-Vid pickle; list entries are named by their zero-padded index."""
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        entries = list(data.items())
    elif isinstance(data, list):
        entries = [(f"{index:04d}", entry) for index, entry in enumerate(data)]
    else:
        raise TypeError(f"{path}: expected a dict or a list of TAP-Vid entries, got {type(data).__name__}")
    for name, entry in entries:
        missing = [key for key in REQUIRED_KEYS if key not in entry]
        if missing:
            raise ValueError(f"{path}: entry {name} is missing {missing}")
    return entries


def decode_video(video) -> np.ndarray:
    """[T, H, W, 3] uint8 RGB from an array or a sequence of JPEG/PNG bytes."""
    if isinstance(video, np.ndarray) and video.ndim == 4:
        return video
    if isinstance(video, (list, tuple)) and video and isinstance(video[0], bytes):
        frames = [cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR) for frame in video]
        return np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    raise ValueError(f"unsupported TAP-Vid video of type {type(video).__name__}")


def make_sample(name: str, entry: dict, size_hw: tuple[int, int]) -> TapVidSample:
    height, width = size_hw
    video = resize_frames(decode_video(entry["video"]), size_hw)
    occluded = np.asarray(entry["occluded"], dtype=bool)
    keep = (~occluded).sum(axis=1) > 1
    occluded = occluded[keep]
    points = np.asarray(entry["points"], dtype=np.float32)[keep] * np.array([width, height], dtype=np.float32)
    first = np.argmax(~occluded, axis=1)
    xy = points[np.arange(len(points)), first]
    query_points = np.stack([first.astype(np.float32), xy[:, 1], xy[:, 0]], axis=1)
    return TapVidSample(name=name, video=video, points=points, occluded=occluded, query_points=query_points)


class TapVidDataset(torch.utils.data.Dataset):
    """TAP-Vid pickle as a map-style dataset of ``TapVidSample`` resized to ``size_hw``."""

    def __init__(self, pickle_path: str | Path, size_hw: tuple[int, int]):
        self.entries = read_tapvid_pickle(pickle_path)
        self.size_hw = size_hw

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> TapVidSample:
        name, entry = self.entries[index]
        return make_sample(name, entry, self.size_hw)
