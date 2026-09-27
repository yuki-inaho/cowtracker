"""``cow-video``: AllTracker-style dense track video from a ``cow-track`` npz and its source."""

import mediapy
import numpy as np
import pytest
from fakes import TranslatingTracker
from PIL import Image

from cowtracker.toolkit import track_cli, video_cli


def _tracked(tmp_path):
    source = tmp_path / "rgb"
    source.mkdir()
    for index in range(6):
        Image.fromarray(np.full((56, 112, 3), 30 * index, np.uint8)).save(source / f"frame_{index:06d}.png")
    track_cli.main(
        [
            "--source",
            str(source),
            "--size",
            "112",
            "224",
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "out"),
            "--query-frame",
            "1",
        ],
        tracker=TranslatingTracker(),
    )
    return source, tmp_path / "out" / "tracks.npz"


@pytest.mark.parametrize(("stack", "shape"), [("h", (112, 448)), ("v", (224, 224)), ("none", (112, 224))])
def test_video_cli_renders_dense_tracks_with_input_stacked(tmp_path, stack, shape):
    source, tracks = _tracked(tmp_path)
    out = tmp_path / f"dense_{stack}.mp4"
    video_cli.main(
        [
            "--tracks",
            str(tracks),
            "--source",
            str(source),
            "--out",
            str(out),
            "--rate",
            "4",
            "--stack",
            stack,
            "--bkg-opacity",
            "0.0",
            "--fps",
            "8",
        ]
    )

    video = mediapy.read_video(str(out))
    assert video.shape == (6, *shape, 3)
    assert video.metadata.fps == pytest.approx(8.0)


def test_video_cli_rejects_mismatched_source(tmp_path):
    source, tracks = _tracked(tmp_path)
    for extra in source.glob("frame_00000[3-5].png"):
        extra.unlink()

    with pytest.raises(ValueError):
        video_cli.main(["--tracks", str(tracks), "--source", str(source), "--out", str(tmp_path / "x.mp4")])
