"""Rendered point-track videos in the style of AllTracker's ``demo.py``: every ``rate``-th pixel's track splatted
with a soft round icon, position-dependent (Bremm) colours, a dimmed background (or black with boosted saturation)."""

from pathlib import Path

import cv2
import mediapy
import numpy as np
import torch

from cowtracker.toolkit.tracks import DenseTracks
from cowtracker.utils.visualization import get_2d_colors

# rate -> icon radius in pixels, as in AllTracker's draw_pts_gpu
RADIUS = {1: 1, 2: 1, 4: 2, 8: 4}


def grid_tracks(
    dense: DenseTracks, stride: int, vis_thr: float, query_frame: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every ``stride``-th pixel of the query frame: tracks [P, T, 2], visible [P, T] (visconf > vis_thr) and
    colours [P, 3] from the pixel's position in the query frame."""
    frames, height, width = dense.vis.shape
    tracks = dense.track[:, ::stride, ::stride].reshape(frames, -1, 2).transpose(1, 0, 2)
    visible = (dense.visconf[:, ::stride, ::stride] > vis_thr).reshape(frames, -1).T
    colors = get_2d_colors(tracks[:, query_frame], height, width)
    return np.ascontiguousarray(tracks, dtype=np.float32), np.ascontiguousarray(visible), colors


@torch.no_grad()
def splat_tracks(
    frames: np.ndarray,
    tracks: np.ndarray,
    visible: np.ndarray,
    colors: np.ndarray,
    rate: int,
    bkg_opacity: float = 0.5,
    device: str | None = None,
) -> np.ndarray:
    """Frames [T, H, W, 3] uint8 with the visible points of tracks [P, T, 2] drawn in colours [P, 3].

    Port of AllTracker's ``draw_pts_gpu`` (MIT): the background is scaled by ``bkg_opacity``; with 0 the dots are
    drawn on black and the saturation is boosted x1.5. ``device`` defaults to CUDA when available (pure drawing).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    video = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float() * bkg_opacity  # T, 3, H, W
    num_frames, channels, height, width = video.shape
    points = torch.from_numpy(np.ascontiguousarray(tracks, dtype=np.float32)).to(device)
    shown = torch.from_numpy(np.ascontiguousarray(visible, dtype=bool)).to(device)
    palette = torch.as_tensor(np.asarray(colors), dtype=torch.float32, device=device)

    radius = RADIUS.get(rate, 6)
    opacity = 0.9 if rate == 1 else 1.0
    sharpness = 0.15 + 0.05 * np.log2(rate)
    offsets = torch.arange(-radius, radius + 1, device=device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    icon = torch.clamp(1 - ((dx**2 + dy**2).float() - radius**2 / 2.0) / (radius * 2 * sharpness), 0, 1)

    for t in range(num_frames):
        mask = shown[:, t]
        if not bool(mask.any()):
            continue
        xy = points[mask, t] + 0.5
        cx = xy[:, 0].clamp(0, width - 1).long()
        cy = xy[:, 1].clamp(0, height - 1).long()
        x_grid, y_grid = cx[:, None, None] + dx, cy[:, None, None] + dy  # P, D, D
        inside = (x_grid >= 0) & (x_grid < width) & (y_grid >= 0) & (y_grid < height)
        weights = icon.expand(len(xy), -1, -1)[inside]
        pixel = (y_grid[inside] * width + x_grid[inside]).long()
        point_colors = palette[mask][:, :, None, None].expand(-1, -1, *icon.shape).permute(1, 0, 2, 3)[:, inside]
        accum = torch.zeros(channels, height * width, device=device)
        accum.scatter_add_(1, pixel.expand(channels, -1), point_colors * weights)
        weight = torch.zeros(1, height * width, device=device).scatter_add_(1, pixel[None], weights[None])
        alpha = (weight.clamp(0, 1) * opacity).view(1, height, width)
        video[t] = video[t] * (1 - alpha) + (accum / (weight + 1e-6)).view(channels, height, width) * alpha

    out = video.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    if bkg_opacity == 0.0:
        for t in range(num_frames):
            hsv = cv2.cvtColor(out[t], cv2.COLOR_RGB2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
            out[t] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return out


def paint_tracks(
    frames: np.ndarray, tracks: np.ndarray, visible: np.ndarray, colors: np.ndarray, rate: int
) -> np.ndarray:
    """Tracks drawn on the frames dimmed to half (the look of the upstream and AllTracker demos)."""
    return splat_tracks(frames, tracks, visible, colors, rate=rate, bkg_opacity=0.5)


def write_video(path: str | Path, frames: np.ndarray, fps: float, crf: float | None = None) -> None:
    mediapy.write_video(str(path), frames, fps=fps, crf=crf)
