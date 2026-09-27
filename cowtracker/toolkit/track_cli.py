"""``cow-track``: dense point tracking of a video file or an image directory.

Writes ``<out>/tracks.npz`` (tracks [T, H, W, 2] float32 (x, y) px of every pixel of the first frame, vis and conf
[T, H, W] float16, frame_ids, size_hw, source_hw), ``<out>/tracks.mp4`` (a grid of tracks drawn on the frames) and
``<out>/meta.json`` (arguments, mode per call, time, peak VRAM, checkpoint hash, git commit, torch version) and, with
``--rrd``, ``<out>/tracks.rrd`` for Rerun.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cowtracker.toolkit import render, runtime
from cowtracker.toolkit.query_tracking import dense_tracks_from
from cowtracker.toolkit.rerun_cli import export_rrd
from cowtracker.toolkit.runner import MODES, build_runner, validate_size
from cowtracker.toolkit.tracks import DenseTracker
from cowtracker.toolkit.video_io import load_frames, resize_frames

DEFAULT_IMAGE_FPS = 10.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cow-track", description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="video file (.mp4 .avi .mov .mkv .webm) or image directory")
    parser.add_argument("--start", type=int, default=0, help="first frame position (0-based, in sorted order)")
    parser.add_argument("--end", type=int, default=None, help="end position (exclusive; default: all frames)")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--query-frame",
        type=int,
        default=0,
        help="track the pixels of this frame (index within the selected frames) forward and backward",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[336, 560],
        metavar=("H", "W"),
        help="inference size; H and W must be multiples of 112 (336 560 for ~16:9, 336 448 for 4:3)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="cowtracker_model.pth (default: download facebook/cowtracker from the Hugging Face hub)",
    )
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu; never switched implicitly")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="auto",
        help="auto: one pass up to --window-len frames, the windowed model beyond",
    )
    parser.add_argument("--window-len", type=int, default=100, help="window of the windowed model (upstream: 100)")
    parser.add_argument(
        "--vram-limit-gb",
        type=float,
        default=None,
        help="per-process VRAM cap; refuses to start when less than this is free",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--grid-stride", type=int, default=8, help="draw every n-th pixel's track")
    parser.add_argument(
        "--vis-thr",
        type=float,
        default=0.1,
        help="display threshold on vis*conf (upstream demo: 0.1; the TAP-Vid evaluation uses 0.6)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=f"output fps (default: the source fps; {DEFAULT_IMAGE_FPS} for image directories)",
    )
    parser.add_argument("--rrd", action="store_true", help="also write <out>/tracks.rrd (Rerun), see cow-rerun")
    return parser


def main(argv: list[str] | None = None, tracker: DenseTracker | None = None) -> int:
    args = build_parser().parse_args(argv)
    size_hw = validate_size(tuple(args.size))
    frames = load_frames(args.source, args.start, args.end, args.step, args.max_frames)
    rgb = resize_frames(frames.rgb, size_hw)
    checkpoint = None
    if tracker is None:
        tracker, checkpoint = build_runner(args.checkpoint, args.device, args.vram_limit_gb, args.window_len, args.mode)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    lengths = []

    def counted(video: np.ndarray):
        lengths.append(len(video))
        return tracker(video)

    start = time.perf_counter()
    dense = dense_tracks_from(counted, rgb, args.query_frame)
    seconds = time.perf_counter() - start

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "tracks.npz",
        tracks=dense.track.astype(np.float32),
        vis=dense.vis.astype(np.float16),
        conf=dense.conf.astype(np.float16),
        frame_ids=frames.frame_ids,
        size_hw=np.array(size_hw),
        source_hw=np.array(frames.rgb.shape[1:3]),
        query_frame=args.query_frame,
    )
    tracks, visible, colors = render.grid_tracks(dense, args.grid_stride, args.vis_thr, args.query_frame)
    fps = args.fps or frames.fps or DEFAULT_IMAGE_FPS
    render.write_video(out / "tracks.mp4", render.paint_tracks(rgb, tracks, visible, colors, args.grid_stride), fps)
    if args.rrd:
        export_rrd(
            out / "tracks.rrd",
            rgb,
            dense,
            frames.frame_ids,
            args.grid_stride,
            trail=8,
            jpeg_quality=85,
            vis_thr=args.vis_thr,
            query_frame=args.query_frame,
        )
    meta = {
        "args": vars(args),
        "frames": len(rgb),
        "fps": fps,
        "calls": getattr(tracker, "calls", [{"frames": n, "mode": "injected"} for n in lengths]),
        "dtype": str(getattr(tracker, "dtype", None)),
        "seconds": seconds,
        "peak_vram_gb": runtime.peak_vram_gb() if args.device.startswith("cuda") else None,
        "checkpoint": runtime.checkpoint_info(checkpoint),
        **runtime.run_metadata(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out}: {len(rgb)} frames at {size_hw[0]}x{size_hw[1]} in {seconds:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
