"""Rendered point-track videos: a grid of dense tracks painted with upstream ``paint_point_track`` and
position-dependent (Bremm) colours as in AllTracker's demo."""

from pathlib import Path

import mediapy
import numpy as np

from cowtracker.toolkit.tracks import DenseTracks
from cowtracker.utils.visualization import get_2d_colors, paint_point_track


def grid_tracks(dense: DenseTracks, stride: int, vis_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every ``stride``-th pixel of frame 0: tracks [P, T, 2], visible [P, T] (visconf > vis_thr), colours [P, 3]."""
    frames, height, width = dense.vis.shape
    tracks = dense.track[:, ::stride, ::stride].reshape(frames, -1, 2).transpose(1, 0, 2)
    visible = (dense.visconf[:, ::stride, ::stride] > vis_thr).reshape(frames, -1).T
    colors = get_2d_colors(tracks[:, 0], height, width)
    return np.ascontiguousarray(tracks, dtype=np.float32), np.ascontiguousarray(visible), colors


def paint_tracks(
    frames: np.ndarray, tracks: np.ndarray, visible: np.ndarray, colors: np.ndarray, rate: int
) -> np.ndarray:
    """Frames [T, H, W, 3] uint8 with the visible points drawn (``rate`` sets the dot size as upstream)."""
    return paint_point_track(
        frames, np.ascontiguousarray(tracks, dtype=np.float32), np.ascontiguousarray(visible), colors, rate=rate
    )


def write_video(path: str | Path, frames: np.ndarray, fps: float) -> None:
    mediapy.write_video(str(path), frames, fps=fps)
