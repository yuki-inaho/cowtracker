"""``cow-track``: frames in, dense tracks (npz), a rendered mp4 and run metadata out."""

import json

import numpy as np
import pytest
from fakes import TranslatingTracker
from PIL import Image
from test_rerun_log import entity_paths

from cowtracker.toolkit import track_cli


def _image_dir(tmp_path, count=6):
    directory = tmp_path / "rgb"
    directory.mkdir()
    for index in range(count):
        Image.fromarray(np.full((60, 80, 3), 40 * index, np.uint8)).save(directory / f"frame_{index:06d}.png")
    return directory


def test_track_cli_writes_npz_mp4_meta(tmp_path):
    out = tmp_path / "out"
    track_cli.main(
        [
            "--source",
            str(_image_dir(tmp_path)),
            "--start",
            "1",
            "--size",
            "112",
            "224",
            "--device",
            "cpu",
            "--out",
            str(out),
            "--grid-stride",
            "16",
        ],
        tracker=TranslatingTracker(),
    )

    with np.load(out / "tracks.npz") as data:
        assert data["tracks"].shape == (5, 112, 224, 2)
        assert data["tracks"].dtype == np.float32
        assert data["vis"].shape == data["conf"].shape == (5, 112, 224)
        assert data["frame_ids"].tolist() == [1, 2, 3, 4, 5]
        assert data["size_hw"].tolist() == [112, 224]
        assert data["source_hw"].tolist() == [60, 80]
        np.testing.assert_allclose(data["tracks"][2, 10, 20], [22.0, 14.0])
    assert (out / "tracks.mp4").stat().st_size > 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["checkpoint"] is None
    assert meta["calls"] == [{"frames": 5, "mode": "injected"}]
    assert meta["frames"] == 5
    assert {"git_commit", "torch", "seconds", "peak_vram_gb", "args"} <= set(meta)


def test_track_cli_rejects_a_size_the_model_cannot_take(tmp_path):
    with pytest.raises(ValueError, match="multiples of 112"):
        track_cli.main(
            [
                "--source",
                str(_image_dir(tmp_path)),
                "--size",
                "100",
                "200",
                "--device",
                "cpu",
                "--out",
                str(tmp_path / "out"),
            ],
            tracker=TranslatingTracker(),
        )


def test_track_cli_writes_rrd_when_requested(tmp_path):
    out = tmp_path / "out"
    track_cli.main(
        ["--source", str(_image_dir(tmp_path)), "--size", "112", "224", "--device", "cpu", "--out", str(out), "--rrd"],
        tracker=TranslatingTracker(),
    )

    assert "/frame/pred/points" in entity_paths(out / "tracks.rrd")


def test_track_cli_tracks_from_a_query_frame_both_ways(tmp_path):
    out = tmp_path / "out"
    tracker = TranslatingTracker()
    track_cli.main(
        [
            "--source",
            str(_image_dir(tmp_path)),
            "--size",
            "112",
            "224",
            "--device",
            "cpu",
            "--out",
            str(out),
            "--query-frame",
            "2",
            "--rrd",
        ],
        tracker=tracker,
    )

    assert tracker.lengths == [4, 3]
    with np.load(out / "tracks.npz") as data:
        assert int(data["query_frame"]) == 2
        np.testing.assert_allclose(data["tracks"][2, 10, 20], [20.0, 10.0])  # identity at the query frame
        np.testing.assert_allclose(data["tracks"][0, 10, 20], [22.0, 14.0])  # two frames before it
