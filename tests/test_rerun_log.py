"""Rerun export of point tracks, checked by loading the .rrd back with ``rerun.dataframe``."""

import numpy as np
import rerun.dataframe as rdf

from cowtracker.toolkit.rerun_log import TrackLayer, write_tracks_rrd

T, H, W, N = 5, 8, 8, 4


def _layer(seed):
    rng = np.random.default_rng(seed)
    tracks = rng.uniform(0, W - 1, (N, T, 2)).astype(np.float32)
    visible = rng.uniform(size=(N, T)) > 0.3
    visible[:, 0] = True
    return TrackLayer(tracks=tracks, visible=visible, colors=rng.integers(0, 255, (N, 3), dtype=np.uint8))


def entity_paths(path):
    return {column.entity_path for column in rdf.load_recording(str(path)).schema().component_columns()}


def test_write_tracks_rrd_roundtrip(tmp_path):
    path = tmp_path / "tracks.rrd"
    frames = np.zeros((T, H, W, 3), dtype=np.uint8)
    write_tracks_rrd(path, frames, {"pred": _layer(0), "gt": _layer(1)}, frame_ids=np.arange(10, 10 + T), trail=3)

    assert {
        "/frame/image",
        "/frame/pred/points",
        "/frame/pred/trails",
        "/frame/gt/points",
        "/metrics/pred/visible_fraction",
        "/metrics/gt/visible_fraction",
    } <= entity_paths(path)
    table = rdf.load_recording(str(path)).view(index="frame", contents="/frame/image").select().read_all()
    assert sorted(table.column("frame").to_pylist()) == list(range(10, 10 + T))
