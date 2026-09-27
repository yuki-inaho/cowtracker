"""Sparse TAP-Vid queries answered by a dense tracker, as in AllTracker's ``test_dense_on_sparse.py``.

For every distinct query frame q the tracker runs on ``video[q:]``; each query reads the dense track of the pixel
nearest to its (x, y). Frames before q keep the query location with visibility-confidence 0 (they are not
evaluated in "first" mode). A query on the last frame has nothing to track and is not run.
"""

import numpy as np

from cowtracker.toolkit.tracks import DenseTracker


def track_queries_first(
    tracker: DenseTracker, video: np.ndarray, query_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Tracks [N, T, 2] (x, y) and visibility-confidence [N, T] of queries [N, 3] = (t, y, x) through video."""
    frames, height, width = video.shape[:3]
    query_frames = np.round(query_points[:, 0]).astype(int)
    xy = query_points[:, [2, 1]].astype(np.float32)
    tracks = np.repeat(xy[:, None], frames, axis=1)
    visconf = np.zeros((len(query_points), frames), dtype=np.float32)
    for query_frame in np.unique(query_frames):
        if query_frame >= frames - 1:
            continue
        rows = np.flatnonzero(query_frames == query_frame)
        dense = tracker(video[query_frame:])
        columns = np.clip(np.round(xy[rows, 0]).astype(int), 0, width - 1)
        lines = np.clip(np.round(xy[rows, 1]).astype(int), 0, height - 1)
        tracks[rows, query_frame:] = dense.track[:, lines, columns].transpose(1, 0, 2)
        visconf[rows, query_frame:] = dense.visconf[:, lines, columns].T
    return tracks, visconf
