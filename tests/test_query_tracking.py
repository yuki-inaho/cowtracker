"""First-mode query tracking on a dense tracker: one run per query frame on the video suffix."""

import numpy as np
from fakes import TranslatingTracker

from cowtracker.toolkit.query_tracking import dense_tracks_from, track_queries_first

T, H, W = 6, 10, 14
STEP = np.array([1.0, 2.0])  # every pixel moves by (+1, +2) px per frame


def _video():
    return np.zeros((T, H, W, 3), dtype=np.uint8)


def test_queries_on_frame0_follow_translation():
    tracker = TranslatingTracker()
    queries = np.array([[0, 3.0, 4.0], [0, 7.2, 1.6]])  # (t, y, x); the second is rounded to pixel (2, 7)
    tracks, visconf = track_queries_first(tracker, _video(), queries)

    assert tracker.lengths == [T]
    np.testing.assert_allclose(tracks[0], np.array([4.0, 3.0]) + np.arange(T)[:, None] * STEP)
    np.testing.assert_allclose(tracks[1], np.array([2.0, 7.0]) + np.arange(T)[:, None] * STEP)
    np.testing.assert_allclose(visconf, 0.5)


def test_query_on_later_frame_runs_on_suffix():
    tracker = TranslatingTracker()
    tracks, visconf = track_queries_first(tracker, _video(), np.array([[3, 5.0, 6.0]]))

    assert tracker.lengths == [T - 3]
    np.testing.assert_allclose(tracks[0, :3], [[6.0, 5.0]] * 3)  # before the query: the query location
    np.testing.assert_allclose(tracks[0, 3:], np.array([6.0, 5.0]) + np.arange(T - 3)[:, None] * STEP)
    np.testing.assert_allclose(visconf[0], [0, 0, 0, 0.5, 0.5, 0.5])


def test_queries_sharing_a_frame_share_one_run():
    tracker = TranslatingTracker()
    track_queries_first(tracker, _video(), np.array([[1, 1.0, 1.0], [1, 2.0, 2.0], [0, 3.0, 3.0]]))

    assert sorted(tracker.lengths) == [T - 1, T]


def test_last_frame_query_does_not_call_tracker():
    tracker = TranslatingTracker()
    tracks, visconf = track_queries_first(tracker, _video(), np.array([[T - 1, 2.0, 3.0]]))

    assert tracker.lengths == []
    np.testing.assert_allclose(tracks[0], [[3.0, 2.0]] * T)
    np.testing.assert_allclose(visconf[0], 0.0)


def test_query_on_the_border_is_clamped_inside_the_frame():
    tracker = TranslatingTracker()
    tracks, _ = track_queries_first(tracker, _video(), np.array([[0, H - 0.4, W - 0.2]]))

    np.testing.assert_allclose(tracks[0, 0], [W - 1, H - 1])


class RecordingTracker(TranslatingTracker):
    """Also records the first frame's value of every call (frame t of the video is filled with t)."""

    def __init__(self):
        super().__init__()
        self.first_values = []

    def __call__(self, video):
        self.first_values.append(int(video[0, 0, 0, 0]))
        return super().__call__(video)


def _numbered_video():
    return np.broadcast_to(np.arange(T, dtype=np.uint8)[:, None, None, None], (T, H, W, 3)).copy()


def test_dense_tracks_from_a_middle_frame_run_both_ways():
    tracker = RecordingTracker()
    dense = dense_tracks_from(tracker, _numbered_video(), query_frame=2)

    assert tracker.lengths == [T - 2, 3]  # forward from frame 2, backward over frames 2, 1, 0
    assert tracker.first_values == [2, 2]
    grid = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), axis=-1)
    assert dense.track.shape == (T, H, W, 2)
    for t in range(T):
        np.testing.assert_allclose(dense.track[t], grid + abs(t - 2) * STEP)


def test_query_frame_zero_runs_forward_only():
    tracker = RecordingTracker()
    dense = dense_tracks_from(tracker, _numbered_video(), query_frame=0)

    assert tracker.lengths == [T]
    assert dense.vis.shape == (T, H, W)


def test_query_frame_outside_the_video_raises():
    import pytest

    with pytest.raises(ValueError, match="query frame"):
        dense_tracks_from(RecordingTracker(), _numbered_video(), query_frame=T)
