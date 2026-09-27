"""``cow-rerun``: an .rrd from a ``cow-track`` npz and its source frames."""

import numpy as np
import pytest
from fakes import TranslatingTracker
from PIL import Image
from test_rerun_log import entity_paths

from cowtracker.toolkit import rerun_cli, track_cli


def _tracked(tmp_path, start=2):
    source = tmp_path / "rgb"
    source.mkdir()
    for index in range(8):
        Image.fromarray(np.full((56, 112, 3), 30 * index, np.uint8)).save(source / f"frame_{index:06d}.png")
    track_cli.main(
        [
            "--source",
            str(source),
            "--start",
            str(start),
            "--size",
            "112",
            "224",
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "out"),
        ],
        tracker=TranslatingTracker(),
    )
    return source, tmp_path / "out" / "tracks.npz"


def test_rerun_cli_from_npz(tmp_path):
    source, tracks = _tracked(tmp_path)
    rerun_cli.main(
        ["--tracks", str(tracks), "--source", str(source), "--out", str(tmp_path / "tracks.rrd"), "--grid-stride", "32"]
    )

    assert {"/frame/image", "/frame/pred/points", "/frame/pred/trails"} <= entity_paths(tmp_path / "tracks.rrd")


def test_rerun_cli_rejects_a_source_that_does_not_match_the_npz(tmp_path):
    source, tracks = _tracked(tmp_path, start=5)
    for extra in source.glob("frame_00000[5-7].png"):
        extra.unlink()

    with pytest.raises(ValueError):
        rerun_cli.main(["--tracks", str(tracks), "--source", str(source), "--out", str(tmp_path / "x.rrd")])
