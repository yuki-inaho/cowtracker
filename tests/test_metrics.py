"""TAP-Vid metrics (tapnet definition, as ported in AllTracker): expected values computed by hand."""

import numpy as np
import pytest

from cowtracker.toolkit.metrics import compute_tapvid_metrics

N, T = 4, 6


def _scene(query_frame=0):
    rng = np.random.default_rng(0)
    gt_tracks = rng.uniform(20, 200, (1, N, T, 2))
    gt_occluded = np.zeros((1, N, T), dtype=bool)
    gt_occluded[0, :, :query_frame] = True
    query_points = np.zeros((1, N, 3))
    query_points[0, :, 0] = query_frame
    query_points[0, :, 1:] = gt_tracks[0, :, query_frame, ::-1]
    return query_points, gt_occluded, gt_tracks


def _metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, crop_size=(256, 256)):
    return compute_tapvid_metrics(
        query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, query_mode="first", crop_size=crop_size
    )


def test_perfect_prediction_scores_one():
    query_points, gt_occluded, gt_tracks = _scene()
    metrics = _metrics(query_points, gt_occluded, gt_tracks, gt_occluded.copy(), gt_tracks.copy())

    assert metrics["average_jaccard"][0] == pytest.approx(1.0)
    assert metrics["average_pts_within_thresh"][0] == pytest.approx(1.0)
    assert metrics["occlusion_accuracy"][0] == pytest.approx(1.0)


def test_offset_within_thresholds():
    query_points, gt_occluded, gt_tracks = _scene()
    pred = gt_tracks + np.array([3.0, 0.0])  # 3 px: outside 1 and 2, inside 4, 8 and 16
    metrics = _metrics(query_points, gt_occluded, gt_tracks, gt_occluded.copy(), pred)

    assert [metrics[f"pts_within_{t}"][0] for t in (1, 2, 4, 8, 16)] == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert metrics["average_pts_within_thresh"][0] == pytest.approx(0.6)
    # thresholds 1 and 2: TP = 0, FP = all predicted-visible points -> Jaccard 0; 4, 8, 16 -> 1
    assert metrics["average_jaccard"][0] == pytest.approx(0.6)


def test_thresholds_are_measured_at_256_scale():
    query_points, gt_occluded, gt_tracks = _scene()
    scale = 511.0 / 255.0  # crop 512 -> (512 - 1) / 255
    pred = (gt_tracks + np.array([3.0, 0.0])) * scale
    metrics = _metrics(query_points, gt_occluded, gt_tracks * scale, gt_occluded.copy(), pred, crop_size=(512, 512))

    assert metrics["average_pts_within_thresh"][0] == pytest.approx(0.6)


def test_first_mode_ignores_frames_before_query():
    query_points, gt_occluded, gt_tracks = _scene(query_frame=2)
    pred = gt_tracks.copy()
    pred[0, :, :2] += 100.0  # wrong before the query frame
    pred_occluded = gt_occluded.copy()
    pred_occluded[0, :, :2] = False  # and wrongly visible there
    metrics = _metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred)

    assert metrics["average_jaccard"][0] == pytest.approx(1.0)
    assert metrics["occlusion_accuracy"][0] == pytest.approx(1.0)


def test_occlusion_accuracy_counts_mismatch():
    query_points, gt_occluded, gt_tracks = _scene()
    pred_occluded = gt_occluded.copy()
    pred_occluded[0, 0, 3] = True  # one of N * (T - 1) evaluated points
    metrics = _metrics(query_points, gt_occluded, gt_tracks, pred_occluded, gt_tracks.copy())

    assert metrics["occlusion_accuracy"][0] == pytest.approx(1.0 - 1.0 / (N * (T - 1)))
