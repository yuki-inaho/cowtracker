"""``cow-rerun``: a Rerun recording (.rrd) of a ``cow-track`` result (``tracks.npz`` + the same source)."""

import argparse
from pathlib import Path

import numpy as np

from cowtracker.toolkit import render
from cowtracker.toolkit.rerun_log import TrackLayer, write_tracks_rrd
from cowtracker.toolkit.tracks import DenseTracks
from cowtracker.toolkit.video_io import load_frames, resize_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cow-rerun", description=__doc__.splitlines()[0])
    parser.add_argument("--tracks", required=True, help="tracks.npz written by cow-track")
    parser.add_argument("--source", required=True, help="the video file or image directory given to cow-track")
    parser.add_argument("--out", required=True, help="output .rrd")
    parser.add_argument("--grid-stride", type=int, default=16, help="log every n-th pixel's track")
    parser.add_argument("--trail", type=int, default=8, help="trail length in frames")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--vis-thr", type=float, default=0.1, help="display threshold on vis*conf (demo: 0.1)")
    return parser


def source_frames(source: str | Path, frame_ids: np.ndarray) -> np.ndarray:
    """The source frames at ``frame_ids`` (an arithmetic range, as written by cow-track)."""
    step = int(frame_ids[1] - frame_ids[0]) if len(frame_ids) > 1 else 1
    frames = load_frames(source, int(frame_ids[0]), int(frame_ids[-1]) + 1, step)
    if not np.array_equal(frames.frame_ids, frame_ids):
        raise ValueError(f"{source} does not provide the frame ids of the tracks")
    return frames.rgb


def export_rrd(
    out: str | Path,
    rgb: np.ndarray,
    dense: DenseTracks,
    frame_ids: np.ndarray,
    grid_stride: int,
    trail: int,
    jpeg_quality: int,
    vis_thr: float,
) -> None:
    tracks, visible, colors = render.grid_tracks(dense, grid_stride, vis_thr)
    write_tracks_rrd(out, rgb, {"pred": TrackLayer(tracks, visible, colors)}, frame_ids, trail, jpeg_quality)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with np.load(args.tracks) as data:
        dense = DenseTracks(
            track=data["tracks"], vis=data["vis"].astype(np.float32), conf=data["conf"].astype(np.float32)
        )
        frame_ids, size_hw = data["frame_ids"], tuple(int(v) for v in data["size_hw"])
    rgb = resize_frames(source_frames(args.source, frame_ids), size_hw)
    export_rrd(args.out, rgb, dense, frame_ids, args.grid_stride, args.trail, args.jpeg_quality, args.vis_thr)
    print(f"wrote {args.out}: {len(frame_ids)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
