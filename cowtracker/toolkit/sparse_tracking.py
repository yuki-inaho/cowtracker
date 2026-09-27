"""Tracks of given query pixels through short image clips, answered by one forward pass of a dense tracker.

A job (``cow_sparse_jobs_v1``) names an image directory, its frames in time order (the first is the birth frame of
every query) and the queries (x, y) in source pixels of the birth frame (integer pixel centres, sub-pixel allowed).
The clip is resized to the inference size, the dense tracks and vis*conf are sampled bilinearly at the queries
mapped to inference pixels, and the tracks are mapped back to source pixels.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from cowtracker.toolkit.tracks import DenseTracker

JOBS_SCHEMA = "cow_sparse_jobs_v1"
JOB_KEYS = {"id", "image_dir", "frames", "queries"}


def rescale_pixels(xy: np.ndarray, from_hw: tuple[int, int], to_hw: tuple[int, int]) -> np.ndarray:
    """Pixel positions (x, y) [..., 2] of an image of ``from_hw`` in the same image resized to ``to_hw``.

    Pixel centres are integers and the image edges stay fixed (``cv2.INTER_LINEAR``): u' = (u + 0.5) * W' / W - 0.5.
    """
    scale = np.array([to_hw[1] / from_hw[1], to_hw[0] / from_hw[0]])
    return (np.asarray(xy, dtype=np.float64) + 0.5) * scale - 0.5


def bilinear_sample(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Samples [N, T, ...] of ``field`` [T, H, W, ...] at pixel positions ``xy`` [N, 2] (x, y).

    A position less than half a pixel beyond the outermost pixel centres (still on the image) reads the edge pixel;
    positions off the image raise ValueError.
    """
    height, width = field.shape[1:3]
    xy = np.asarray(xy, dtype=np.float64)
    on_image = (xy >= -0.5).all(1) & (xy[:, 0] <= width - 0.5) & (xy[:, 1] <= height - 0.5)
    if not on_image.all():
        raise ValueError(f"positions {xy[~on_image].tolist()} are off the {height}x{width} image")
    x, y = np.clip(xy[:, 0], 0, width - 1), np.clip(xy[:, 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    trailing = (1,) * (field.ndim - 3)
    sampled = sum(
        field[:, rows, columns] * weight.reshape(1, -1, *trailing)
        for rows, columns, weight in (
            (y0, x0, (1 - dx) * (1 - dy)),
            (y0, x1, dx * (1 - dy)),
            (y1, x0, (1 - dx) * dy),
            (y1, x1, dx * dy),
        )
    )
    return np.moveaxis(sampled, 0, 1)


def track_queries(
    tracker: DenseTracker, video: np.ndarray, queries: np.ndarray, source_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Tracks uv [N, T, 2] (x, y) in source pixels and vis*conf [N, T] (float32) of ``queries`` [N, 2] (x, y) in
    source pixels of ``video[0]``, from one forward pass of ``tracker`` on ``video`` [T, h, w, 3] (the resized clip)."""
    inference_hw = video.shape[1:3]
    dense = tracker(video)
    if dense.track.shape != (*video.shape[:3], 2):
        raise ValueError(f"tracker returned tracks {dense.track.shape} for a video {video.shape}")
    at = rescale_pixels(queries, source_hw, inference_hw)
    uv = rescale_pixels(bilinear_sample(dense.track, at), inference_hw, source_hw)
    return uv.astype(np.float32), bilinear_sample(dense.visconf, at).astype(np.float32)


@dataclass(frozen=True)
class SparseJob:
    """A validated job: ``queries`` [N, 2] (x, y) in pixels of ``frames[0]``, every frame of ``source_hw``."""

    id: str
    image_dir: Path
    frames: tuple[str, ...]
    queries: np.ndarray
    source_hw: tuple[int, int]

    @property
    def paths(self) -> list[Path]:
        return [self.image_dir / name for name in self.frames]


def _image_hw(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.height, image.width


def _validate_job(job: dict, window_len: int) -> SparseJob:
    if not isinstance(job, dict) or set(job) != JOB_KEYS:
        got = sorted(job) if isinstance(job, dict) else type(job).__name__
        raise ValueError(f"a job must be an object with exactly the keys {sorted(JOB_KEYS)}; got {got}")
    job_id = job["id"]
    if not isinstance(job_id, str) or job_id in ("", ".", "..") or Path(job_id).name != job_id:
        raise ValueError(f"job id {job_id!r} must be a plain file name (it names <out>/<id>.npz)")
    image_dir = Path(job["image_dir"])
    if not image_dir.is_absolute():
        raise ValueError(f"job {job_id!r}: image_dir {str(image_dir)!r} must be an absolute path")
    frames = job["frames"]
    if not isinstance(frames, list) or not 2 <= len(frames) <= window_len:
        raise ValueError(f"job {job_id!r}: frames must list 2 to {window_len} (--window-len) images; got {frames!r}")
    if not all(isinstance(name, str) and Path(name).name == name for name in frames):
        raise ValueError(f"job {job_id!r}: frames must be file names inside image_dir; got {frames!r}")
    paths = [image_dir / name for name in frames]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"job {job_id!r}: missing image(s) {missing}")
    sizes = {_image_hw(path) for path in paths}
    if len(sizes) != 1:
        raise ValueError(f"job {job_id!r}: frames of different sizes (H, W) {sorted(sizes)}")
    (source_hw,) = sizes
    queries = np.asarray(job["queries"], dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 2 or not len(queries):
        raise ValueError(f"job {job_id!r}: queries must be a non-empty list of [x, y]; got shape {queries.shape}")
    height, width = source_hw
    inside = (queries >= 0).all(1) & (queries[:, 0] <= width - 1) & (queries[:, 1] <= height - 1)
    if not inside.all():
        raise ValueError(
            f"job {job_id!r}: queries {queries[~inside].tolist()} are outside the {height}x{width} image "
            f"(pixel centres 0..{width - 1}, 0..{height - 1})"
        )
    return SparseJob(job_id, image_dir, tuple(frames), queries, source_hw)


def validate_jobs(spec: dict, window_len: int) -> list[SparseJob]:
    """Every job of a ``cow_sparse_jobs_v1`` spec, checked (images exist and share a size, 2..window_len frames,
    queries on the birth frame, unique ids) before any inference; the first problem raises ValueError."""
    schema = spec.get("schema") if isinstance(spec, dict) else None
    if schema != JOBS_SCHEMA:
        raise ValueError(f"jobs schema must be {JOBS_SCHEMA!r}; got {schema!r}")
    if not isinstance(spec.get("jobs"), list) or not spec["jobs"]:
        raise ValueError("no jobs: 'jobs' must be a non-empty list")
    jobs = [_validate_job(job, window_len) for job in spec["jobs"]]
    duplicates = sorted(job_id for job_id, count in Counter(job.id for job in jobs).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate job ids {duplicates}")
    return jobs
