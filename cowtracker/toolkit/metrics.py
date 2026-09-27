"""TAP-Vid metrics: Occlusion Accuracy, <δ^x (pts_within_x), Jaccard_x and their averages.

Port of ``compute_tapvid_metrics`` from AllTracker (``utils/misc.py``, MIT), itself from the TAP-Vid reference
implementation (google-deepmind/tapnet, Apache-2.0) with the CoTracker fix of issue #20 and an explicit
``crop_size``: coordinates are measured in pixels of a ``crop_size`` frame and rescaled to the 256x256 frame on
which the thresholds 1, 2, 4, 8 and 16 px are defined, via ``(size - 1) / 255``.
"""

import numpy as np

THRESHOLDS = (1, 2, 4, 8, 16)


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    crop_size: tuple[int, int] = (256, 256),
) -> dict[str, np.ndarray]:
    """Per-video metrics (arrays of length b) of a batch.

    Args:
        query_points: [b, n, 3] as (t, y, x).
        gt_occluded, pred_occluded: [b, n, t] bool, True = occluded.
        gt_tracks, pred_tracks: [b, n, t, 2] as (x, y) in pixels of a ``crop_size`` (H, W) frame.
        query_mode: ``first`` (frames after the query frame are evaluated) or ``strided`` (all but the query frame).
    """
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    if query_mode == "first":
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError(f"unknown query mode {query_mode!r}")
    query_frame = np.round(query_points[..., 0]).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    metrics = {
        "occlusion_accuracy": np.sum(np.equal(pred_occluded, gt_occluded) & evaluation_points, axis=(1, 2))
        / np.sum(evaluation_points)
    }
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    scale = np.array([(crop_size[1] - 1) / 255.0, (crop_size[0] - 1) / 255.0]).reshape(1, 1, 1, 2)
    all_frac_within, all_jaccard = [], []
    for thresh in THRESHOLDS:
        within_dist = np.sum(np.square(pred_tracks / scale - gt_tracks / scale), axis=-1) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = np.sum(is_correct & evaluation_points, axis=(1, 2)) / count_visible_points
        metrics[f"pts_within_{thresh}"] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(is_correct & pred_visible & evaluation_points, axis=(1, 2))
        false_positives = ((~visible) & pred_visible) | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (count_visible_points + false_positives)
        metrics[f"jaccard_{thresh}"] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    metrics["average_pts_within_thresh"] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    return metrics
