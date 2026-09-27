"""Test doubles: a dense tracker whose every pixel translates by a constant step per frame."""

import numpy as np

from cowtracker.toolkit.tracks import DenseTracks


class TranslatingTracker:
    """``DenseTracker`` moving every pixel by ``step`` (x, y) px per frame, with constant vis and conf."""

    def __init__(self, step=(1.0, 2.0), vis=1.0, conf=0.5):
        self.step = np.asarray(step, dtype=np.float32)
        self.vis, self.conf = vis, conf
        self.lengths = []

    def __call__(self, video):
        frames, height, width = video.shape[:3]
        self.lengths.append(frames)
        grid = np.stack(np.meshgrid(np.arange(width), np.arange(height), indexing="xy"), axis=-1).astype(np.float32)
        track = grid[None] + np.arange(frames, dtype=np.float32)[:, None, None, None] * self.step
        ones = np.ones((frames, height, width), dtype=np.float32)
        return DenseTracks(track=track, vis=self.vis * ones, conf=self.conf * ones)
