"""Rerun (.rrd) recordings of point tracks: the frame image, per layer (e.g. ``pred`` and ``gt``) the visible points,
their recent trails, and the visible fraction over time. Open with ``rerun <file>.rrd``."""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


class TrackLayer(NamedTuple):
    """Point tracks [N, T, 2] (x, y) px, visibility [N, T] bool and colours [N, 3] uint8."""

    tracks: np.ndarray
    visible: np.ndarray
    colors: np.ndarray


def _blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial2DView(origin="frame", name="tracks"),
            rrb.TimeSeriesView(origin="metrics", name="visible fraction"),
            row_shares=[4, 1],
        ),
        collapse_panels=True,
    )


def write_tracks_rrd(
    path: str | Path,
    frames: np.ndarray,
    layers: dict[str, TrackLayer],
    frame_ids: np.ndarray,
    trail: int = 8,
    jpeg_quality: int = 85,
    application_id: str = "cowtracker",
) -> None:
    """Write frames [T, H, W, 3] uint8 and the track layers to ``path`` on the ``frame`` timeline (``frame_ids``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream(application_id=application_id)
    recording.save(str(path), default_blueprint=_blueprint())
    for t, frame_id in enumerate(frame_ids):
        recording.set_time("frame", sequence=int(frame_id))
        recording.log("frame/image", rr.Image(frames[t]).compress(jpeg_quality=jpeg_quality))
        for name, layer in layers.items():
            shown = np.flatnonzero(layer.visible[:, t])
            recording.log(
                f"frame/{name}/points", rr.Points2D(layer.tracks[shown, t], colors=layer.colors[shown], radii=1.5)
            )
            if t > 0:  # a trail needs two positions
                strips = [layer.tracks[n, max(0, t - trail) : t + 1] for n in shown]
                recording.log(f"frame/{name}/trails", rr.LineStrips2D(strips, colors=layer.colors[shown], radii=0.5))
            recording.log(f"metrics/{name}/visible_fraction", rr.Scalars(float(layer.visible[:, t].mean())))
    recording.flush(blocking=True)
    recording.disconnect()
