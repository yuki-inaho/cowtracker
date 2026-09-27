"""``cow-eval-tapvid``: TAP-Vid first-mode evaluation of a dense tracker on a benchmark pickle."""

import json
import pickle

import numpy as np
from fakes import TranslatingTracker

from cowtracker.toolkit import eval_cli

T, H, W = 6, 112, 224
STEP = np.array([1.0, 2.0])


def _pickle(tmp_path):
    """Two videos whose GT points move exactly like ``TranslatingTracker`` from integer pixels."""
    entries = {}
    for index, name in enumerate(["bear", "car"]):
        start = np.array([[10.0, 20.0], [50.0 + index, 30.0], [100.0, 60.0]])
        points = start[:, None] + np.arange(T)[None, :, None] * STEP
        occluded = np.zeros((3, T), dtype=bool)
        occluded[2, :2] = True  # queried at frame 2
        entries[name] = {
            "video": np.zeros((T, H, W, 3), np.uint8),
            "points": points / np.array([W, H]),
            "occluded": occluded,
        }
    path = tmp_path / "tapvid_davis.pkl"
    path.write_bytes(pickle.dumps(entries))
    return path


def _run(tmp_path, *extra, tracker):
    out = tmp_path / "eval"
    eval_cli.main(
        [
            "--pkl",
            str(_pickle(tmp_path)),
            "--dataset-name",
            "davis",
            "--size",
            str(H),
            str(W),
            "--device",
            "cpu",
            "--out",
            str(out),
            *extra,
        ],
        tracker=tracker,
    )
    return out, json.loads((out / "summary.json").read_text())


def test_eval_cli_perfect_tracker_scores_one(tmp_path):
    out, summary = _run(tmp_path, "--render-videos", "1", tracker=TranslatingTracker(conf=1.0))

    assert summary["n_videos"] == 2
    assert summary["mean"]["aj"] == summary["mean"]["d_avg"] == summary["mean"]["oa"] == 100.0
    assert summary["paper"] == {"aj": 65.5, "d_avg": 78.0, "oa": 92.1}
    assert summary["delta"]["aj"] == 100.0 - 65.5
    assert len((out / "per_video.jsonl").read_text().splitlines()) == 2
    assert (out / "videos" / "bear.mp4").stat().st_size > 0
    assert (out / "videos" / "bear.rrd").stat().st_size > 0
    assert not (out / "videos" / "car.mp4").exists()


def test_eval_cli_caches_predictions_and_threshold_sweep(tmp_path):
    out, summary = _run(tmp_path, "--vis-thr-sweep", "0.3", "0.7", tracker=TranslatingTracker(conf=0.5))

    with np.load(out / "predictions" / "bear.npz") as cached:
        assert cached["tracks"].shape == (3, T, 2)
        assert cached["visconf"].shape == (3, T)
    assert summary["sweep"]["0.3"]["oa"] == 100.0  # visconf 0.5 counts as visible
    assert summary["sweep"]["0.7"]["oa"] == 0.0  # and as occluded
    assert summary["mean"]["oa"] == 0.0  # default threshold 0.6


def test_eval_cli_passes_the_dtype_to_the_runner(tmp_path, monkeypatch):
    seen = {}

    def fake_build_runner(checkpoint, device, vram_limit_gb, window_len, mode, dtype):
        seen.update(window_len=window_len, dtype=dtype)
        return TranslatingTracker(conf=1.0), None

    monkeypatch.setattr(eval_cli, "build_runner", fake_build_runner)
    eval_cli.main(
        [
            "--pkl",
            str(_pickle(tmp_path)),
            "--dataset-name",
            "davis",
            "--size",
            str(H),
            str(W),
            "--device",
            "cpu",
            "--dtype",
            "bf16",
            "--window-len",
            "40",
            "--out",
            str(tmp_path / "eval"),
        ]
    )

    assert seen == {"window_len": 40, "dtype": "bf16"}
