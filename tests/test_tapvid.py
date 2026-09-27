"""TAP-Vid pickles (DAVIS: dict of videos; RGB-Stacking: list; Kinetics/RoboTAP-style JPEG frames) and query points."""

import pickle

import cv2
import numpy as np
import pytest

from cowtracker.toolkit.tapvid import TapVidDataset, read_tapvid_pickle

T, H, W = 5, 8, 12


def _entry(seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.1, 0.9, (3, T, 2)).astype(np.float32)
    occluded = np.zeros((3, T), dtype=bool)
    occluded[1, :2] = True  # track 1 appears at frame 2
    occluded[2, 1:] = True  # track 2 is visible once only: dropped
    video = np.zeros((T, H, W, 3), dtype=np.uint8)
    video[..., 0] = 200  # red
    return {"video": video, "points": points, "occluded": occluded}


def _dump(path, data):
    path.write_bytes(pickle.dumps(data))
    return path


def test_reads_dict_format_like_davis(tmp_path):
    entries = read_tapvid_pickle(_dump(tmp_path / "davis.pkl", {"bear": _entry(), "car": _entry(1)}))

    assert [name for name, _ in entries] == ["bear", "car"]


def test_reads_list_format_like_rgb_stacking(tmp_path):
    entries = read_tapvid_pickle(_dump(tmp_path / "rgbs.pkl", [_entry(), _entry(1)]))

    assert [name for name, _ in entries] == ["0000", "0001"]


def test_decodes_jpeg_bytes_frames(tmp_path):
    entry = _entry()
    entry["video"] = [
        cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes() for frame in entry["video"]
    ]
    sample = TapVidDataset(_dump(tmp_path / "kinetics.pkl", [entry]), size_hw=(H, W))[0]

    assert sample.video.shape == (T, H, W, 3)
    assert abs(int(sample.video[0, 4, 6, 0]) - 200) <= 3  # red stays in channel 0 (RGB, not BGR)
    assert int(sample.video[0, 4, 6, 2]) <= 3


def test_sample_scales_points_to_resized_frame_and_builds_first_queries(tmp_path):
    entry = _entry()
    sample = TapVidDataset(_dump(tmp_path / "davis.pkl", {"bear": entry}), size_hw=(2 * H, 3 * W))[0]

    assert sample.name == "bear"
    assert sample.video.shape == (T, 2 * H, 3 * W, 3)
    kept = entry["points"][:2]  # track 2 has a single visible frame
    np.testing.assert_allclose(sample.points, kept * np.array([3 * W, 2 * H], dtype=np.float32))
    np.testing.assert_array_equal(sample.occluded, entry["occluded"][:2])
    assert sample.query_points[:, 0].tolist() == [0, 2]
    np.testing.assert_allclose(sample.query_points[1, 1:], sample.points[1, 2, ::-1])  # (y, x) at frame 2


def test_missing_keys_raise(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        read_tapvid_pickle(_dump(tmp_path / "bad.pkl", [{"video": np.zeros((T, H, W, 3), np.uint8)}]))


def test_unknown_container_raises(tmp_path):
    with pytest.raises(TypeError, match="dict or a list"):
        read_tapvid_pickle(_dump(tmp_path / "bad.pkl", 3))
