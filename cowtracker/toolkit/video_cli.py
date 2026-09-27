"""``cow-video``: a dense point-track video of a ``cow-track`` result, in the style of AllTracker's ``demo.py``.

Every ``--rate``-th pixel of the query frame is drawn where it is tracked in each frame (visible when vis*conf >
``--conf-thr``), coloured by its position in the query frame, over the frame dimmed by ``--bkg-opacity`` (0: dots on
black with boosted saturation); ``--stack h|v`` puts the input next to or above the result.
"""

import argparse

import numpy as np

from cowtracker.toolkit import render
from cowtracker.toolkit.rerun_cli import load_tracks, source_frames
from cowtracker.toolkit.track_cli import DEFAULT_IMAGE_FPS
from cowtracker.toolkit.video_io import resize_frames

STACK_AXIS = {"h": 2, "v": 1}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cow-video", description=__doc__.splitlines()[0])
    parser.add_argument("--tracks", required=True, help="tracks.npz written by cow-track")
    parser.add_argument("--source", required=True, help="the video file or image directory given to cow-track")
    parser.add_argument("--out", required=True, help="output .mp4")
    parser.add_argument("--rate", type=int, default=2, help="draw every n-th pixel's track (AllTracker demo: 2)")
    parser.add_argument("--conf-thr", type=float, default=0.1, help="visible when vis*conf exceeds this")
    parser.add_argument("--bkg-opacity", type=float, default=0.5, help="background brightness (0: dots on black)")
    parser.add_argument("--stack", choices=["none", "h", "v"], default="none", help="input left of (h) or above (v)")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=f"output fps (default: the source fps; {DEFAULT_IMAGE_FPS} for image directories)",
    )
    parser.add_argument("--crf", type=float, default=20.0, help="x264 quality (AllTracker demo: 20)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dense, frame_ids, size_hw, query_frame = load_tracks(args.tracks)
    frames = source_frames(args.source, frame_ids)
    rgb = resize_frames(frames.rgb, size_hw)
    tracks, visible, colors = render.grid_tracks(dense, args.rate, args.conf_thr, query_frame)
    video = render.splat_tracks(rgb, tracks, visible, colors, rate=args.rate, bkg_opacity=args.bkg_opacity)
    if args.stack != "none":
        video = np.concatenate([rgb, video], axis=STACK_AXIS[args.stack])
    fps = args.fps or frames.fps or DEFAULT_IMAGE_FPS
    render.write_video(args.out, video, fps, crf=args.crf)
    print(f"wrote {args.out}: {len(video)} frames, {len(tracks)} tracks (rate {args.rate}, query frame {query_frame})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
