"""Frame sources for inference: a video file or a directory of images, returned as RGB uint8 frames."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapy
import numpy as np
from PIL import Image

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Frames:
    """Frames [T, H, W, 3] uint8, their positions ``frame_ids`` in the source, and the source fps (None for images)."""

    rgb: np.ndarray
    frame_ids: np.ndarray
    fps: float | None
    source: str


def select_positions(
    count: int, start: int = 0, end: int | None = None, step: int = 1, max_frames: int | None = None
) -> np.ndarray:
    """Positions ``start, start+step, ... < end`` (end exclusive, default ``count``), truncated to ``max_frames``."""
    end = count if end is None else end
    if not 0 <= start < end <= count or step < 1:
        raise ValueError(f"frame range start={start} end={end} step={step} is outside a source of {count} frames")
    positions = np.arange(start, end, step)
    return positions if max_frames is None else positions[:max_frames]


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"no images ({', '.join(IMAGE_SUFFIXES)}) in {directory}")
    return paths


def load_frames(
    source: str | Path, start: int = 0, end: int | None = None, step: int = 1, max_frames: int | None = None
) -> Frames:
    """Load a frame range of a video file (``VIDEO_SUFFIXES``) or an image directory (sorted by file name)."""
    path = Path(source)
    if path.is_dir():
        paths = image_paths(path)
        positions = select_positions(len(paths), start, end, step, max_frames)
        rgb = np.stack([np.asarray(Image.open(paths[i]).convert("RGB")) for i in positions])
        return Frames(rgb=rgb, frame_ids=positions, fps=None, source=str(path))
    if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file():
        video = mediapy.read_video(str(path))
        positions = select_positions(len(video), start, end, step, max_frames)
        return Frames(
            rgb=np.asarray(video)[positions], frame_ids=positions, fps=float(video.metadata.fps), source=str(path)
        )
    raise ValueError(f"unsupported source {path}: expected an image directory or a video file {VIDEO_SUFFIXES}")


def resize_frames(rgb: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of [T, H, W, 3] uint8 frames to ``size_hw`` (as the TAP-Vid readers of AllTracker do)."""
    height, width = size_hw
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in rgb])
