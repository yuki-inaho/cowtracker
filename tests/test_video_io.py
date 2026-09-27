"""Frame sources: video files and image directories."""

import mediapy
import numpy as np
import pytest

from cowtracker.toolkit.video_io import load_frames


def _write_mp4(path, frames=12, fps=24, size=(32, 48)):
    video = np.zeros((frames, *size, 3), dtype=np.uint8)
    video[..., 0] = np.arange(frames, dtype=np.uint8)[:, None, None] * 20
    mediapy.write_video(path, video, fps=fps)
    return path


def test_load_mp4_reads_all_frames_and_fps(tmp_path):
    frames = load_frames(_write_mp4(tmp_path / "clip.mp4"))

    assert frames.rgb.shape == (12, 32, 48, 3)
    assert frames.rgb.dtype == np.uint8
    assert frames.fps == pytest.approx(24.0)
    assert frames.frame_ids.tolist() == list(range(12))


def test_load_mp4_frame_range_and_step(tmp_path):
    frames = load_frames(_write_mp4(tmp_path / "clip.mp4"), start=2, end=10, step=2)

    assert frames.frame_ids.tolist() == [2, 4, 6, 8]
    assert frames.rgb.shape[0] == 4


def test_unknown_suffix_raises(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("not a video")

    with pytest.raises(ValueError, match="unsupported source"):
        load_frames(source)


def _write_pngs(directory, count=12, size=(16, 24)):
    from PIL import Image

    directory.mkdir()
    for index in range(count):
        image = np.full((*size, 3), index * 10, dtype=np.uint8)
        Image.fromarray(image).save(directory / f"frame_{index:06d}.png")
    return directory


def test_load_image_dir_sorted_by_name(tmp_path):
    frames = load_frames(_write_pngs(tmp_path / "rgb"), start=1, end=12, step=5)

    assert frames.frame_ids.tolist() == [1, 6, 11]
    assert frames.rgb[:, 0, 0, 0].tolist() == [10, 60, 110]
    assert frames.fps is None


def test_empty_image_dir_raises(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="no images"):
        load_frames(tmp_path / "empty")


def test_range_beyond_the_source_raises(tmp_path):
    with pytest.raises(ValueError, match="frame range"):
        load_frames(_write_pngs(tmp_path / "rgb"), start=5, end=20)


def test_max_frames_truncates_after_the_step(tmp_path):
    frames = load_frames(_write_pngs(tmp_path / "rgb"), step=2, max_frames=3)

    assert frames.frame_ids.tolist() == [0, 2, 4]


def test_resize_frames_shape_and_dtype():
    from cowtracker.toolkit.video_io import resize_frames

    resized = resize_frames(np.zeros((3, 20, 30, 3), dtype=np.uint8), (112, 224))

    assert resized.shape == (3, 112, 224, 3)
    assert resized.dtype == np.uint8
