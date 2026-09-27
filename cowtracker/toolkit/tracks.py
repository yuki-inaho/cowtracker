"""Dense tracker output and the tracker interface the toolkit depends on."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DenseTracks:
    """Tracks of every pixel of the first frame: ``track`` [T, H, W, 2] (x, y) in pixels, ``vis`` and ``conf``
    [T, H, W] in [0, 1]."""

    track: np.ndarray
    vis: np.ndarray
    conf: np.ndarray

    @property
    def visconf(self) -> np.ndarray:
        """Visibility times confidence, the score thresholded by the demo (0.1) and the evaluation (0.6)."""
        return self.vis * self.conf


class DenseTracker(Protocol):
    def __call__(self, video: np.ndarray) -> DenseTracks:
        """Track all pixels of ``video[0]`` through ``video`` ([T, H, W, 3] uint8)."""
