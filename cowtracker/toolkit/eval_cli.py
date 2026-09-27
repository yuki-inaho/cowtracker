"""``cow-eval-tapvid``: TAP-Vid evaluation ("first" queries) of CoWTracker on a benchmark pickle.

Protocol of AllTracker's ``test_dense_on_sparse.py``: frames resized to ``--size``; for every distinct query frame
the tracker runs on the video suffix and each query reads the dense track at its nearest pixel; a point is predicted
visible when vis*conf >= ``--vis-thr`` (0.6); AJ, <δ^x_avg (d_avg) and OA are averaged over videos (x100).

Outputs in ``--out``: ``per_video.jsonl`` (written as the run goes), ``predictions/<video>.npz`` (tracks, visconf),
``summary.json`` (means, the paper's numbers and differences, optional threshold sweep, provenance) and, for the
first ``--render-videos`` videos, ``videos/<video>.mp4`` (GT | prediction) and ``videos/<video>.rrd``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cowtracker.toolkit import render, runtime
from cowtracker.toolkit.metrics import THRESHOLDS, compute_tapvid_metrics
from cowtracker.toolkit.query_tracking import track_queries_first
from cowtracker.toolkit.rerun_log import TrackLayer, write_tracks_rrd
from cowtracker.toolkit.runner import DTYPES, MODES, build_runner, validate_size
from cowtracker.toolkit.tapvid import TapVidDataset, TapVidSample
from cowtracker.toolkit.tracks import DenseTracker
from cowtracker.utils.visualization import get_2d_colors

# CoWTracker (Kubric), Table 1 of the paper: AJ, <δ^x_avg, OA.
PAPER = {
    "davis": {"aj": 65.5, "d_avg": 78.0, "oa": 92.1},
    "rgb_stacking": {"aj": 85.4, "d_avg": 92.8, "oa": 94.9},
    "robotap": {"aj": 73.2, "d_avg": 83.4, "oa": 94.7},
    "kinetics": {"aj": 60.9, "d_avg": 73.1, "oa": 91.5},
}
# Pre-registered judgement (workdoc): sanity, and the band within which the paper's numbers count as reproduced.
SANITY_D_AVG = 70.0
REPRODUCED_BAND = {"aj": 2.0, "d_avg": 1.5, "oa": 2.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cow-eval-tapvid", description=__doc__.splitlines()[0])
    parser.add_argument("--pkl", required=True, help="TAP-Vid pickle (e.g. tapvid_davis.pkl)")
    parser.add_argument("--dataset-name", required=True, help=f"name in the summary; paper numbers for {list(PAPER)}")
    parser.add_argument("--size", type=int, nargs=2, default=[336, 560], metavar=("H", "W"))
    parser.add_argument("--vis-thr", type=float, default=0.6, help="visible when vis*conf >= this (AllTracker: 0.6)")
    parser.add_argument(
        "--vis-thr-sweep",
        type=float,
        nargs="*",
        default=[],
        help="extra thresholds reported descriptively from the cached predictions",
    )
    parser.add_argument("--max-videos", type=int, default=None, help="first n videos (in pickle order)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-videos", type=int, default=0, help="write mp4 and rrd for the first n videos")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=MODES, default="auto")
    parser.add_argument("--dtype", choices=["auto", *DTYPES], default="auto", help="auto: fp16 on CUDA, fp32 on CPU")
    parser.add_argument("--window-len", type=int, default=100)
    parser.add_argument("--vram-limit-gb", type=float, default=None)
    return parser


def score(sample: TapVidSample, tracks: np.ndarray, visconf: np.ndarray, vis_thr: float) -> dict[str, float]:
    """TAP-Vid metrics (x100) of one video at one visibility threshold."""
    metrics = compute_tapvid_metrics(
        sample.query_points[None],
        sample.occluded[None],
        sample.points[None],
        (visconf < vis_thr)[None],
        tracks[None],
        query_mode="first",
        crop_size=sample.video.shape[1:3],
    )
    result = {
        "aj": metrics["average_jaccard"][0],
        "d_avg": metrics["average_pts_within_thresh"][0],
        "oa": metrics["occlusion_accuracy"][0],
    }
    for thresh in THRESHOLDS:
        result[f"d_{thresh}"] = metrics[f"pts_within_{thresh}"][0]
        result[f"jac_{thresh}"] = metrics[f"jaccard_{thresh}"][0]
    return {key: float(value) * 100.0 for key, value in result.items()}


def mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}


def judge(mean: dict[str, float], paper: dict[str, float] | None) -> dict:
    verdict = {"sanity_d_avg_ge_70": mean["d_avg"] >= SANITY_D_AVG}
    if paper is not None:
        verdict["reproduced"] = all(abs(mean[key] - paper[key]) <= band for key, band in REPRODUCED_BAND.items())
    return verdict


def render_video(
    out: Path, sample: TapVidSample, tracks: np.ndarray, visconf: np.ndarray, vis_thr: float, fps: float = 12.0
) -> None:
    """``out/<name>.mp4`` (GT left, prediction right) and ``out/<name>.rrd`` (layers gt and pred)."""
    height, width = sample.video.shape[1:3]
    colors = get_2d_colors(sample.query_points[:, [2, 1]], height, width)
    gt = TrackLayer(sample.points, ~sample.occluded, colors)
    pred = TrackLayer(tracks, visconf >= vis_thr, colors)
    panels = [render.paint_tracks(sample.video, layer.tracks, layer.visible, colors, rate=8) for layer in (gt, pred)]
    out.mkdir(parents=True, exist_ok=True)
    render.write_video(out / f"{sample.name}.mp4", np.concatenate(panels, axis=2), fps)
    write_tracks_rrd(
        out / f"{sample.name}.rrd",
        sample.video,
        {"gt": gt, "pred": pred},
        np.arange(len(sample.video)),
        trail=8,
        jpeg_quality=85,
    )


def main(argv: list[str] | None = None, tracker: DenseTracker | None = None) -> int:
    args = build_parser().parse_args(argv)
    size_hw = validate_size(tuple(args.size))
    dataset = TapVidDataset(args.pkl, size_hw)
    count = len(dataset) if args.max_videos is None else min(args.max_videos, len(dataset))
    checkpoint = None
    if tracker is None:
        tracker, checkpoint = build_runner(
            args.checkpoint, args.device, args.vram_limit_gb, args.window_len, args.mode, args.dtype
        )
    cuda = args.device.startswith("cuda")
    out = Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    thresholds = [args.vis_thr, *args.vis_thr_sweep]
    rows: dict[float, list[dict]] = {thr: [] for thr in thresholds}
    peaks, start_all = [], time.perf_counter()
    with open(out / "per_video.jsonl", "w") as per_video:
        for index in range(count):
            sample = dataset[index]
            calls_before = len(getattr(tracker, "calls", []))
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            tracks, visconf = track_queries_first(tracker, sample.video, sample.query_points)
            seconds = time.perf_counter() - start
            peaks.append(runtime.peak_vram_gb() if cuda else None)
            np.savez(out / "predictions" / f"{sample.name}.npz", tracks=tracks, visconf=visconf)
            for thr in thresholds:
                rows[thr].append(score(sample, tracks, visconf, thr))
            modes = [call["mode"] for call in getattr(tracker, "calls", [])[calls_before:]]
            record = {
                "name": sample.name,
                "frames": len(sample.video),
                "tracks": len(sample.query_points),
                "query_frames": len(np.unique(np.round(sample.query_points[:, 0]))),
                "calls": {mode: modes.count(mode) for mode in sorted(set(modes))},
                "seconds": seconds,
                "peak_vram_gb": peaks[-1],
                **rows[args.vis_thr][-1],
            }
            per_video.write(json.dumps(record) + "\n")
            per_video.flush()
            print(
                f"[{index + 1}/{count}] {sample.name}: aj {record['aj']:.1f} d_avg {record['d_avg']:.1f} "
                f"oa {record['oa']:.1f} ({seconds:.1f} s)",
                flush=True,
            )
            if index < args.render_videos:
                render_video(out / "videos", sample, tracks, visconf, args.vis_thr)

    mean = mean_scores(rows[args.vis_thr])
    paper = PAPER.get(args.dataset_name)
    measured_peaks = [peak for peak in peaks if peak is not None]
    summary = {
        "dataset": args.dataset_name,
        "n_videos": count,
        "nan_videos": [dataset.entries[i][0] for i, row in enumerate(rows[args.vis_thr]) if np.isnan(row["aj"])],
        "size_hw": list(size_hw),
        "vis_thr": args.vis_thr,
        "mean": mean,
        "paper": paper,
        "delta": None if paper is None else {key: mean[key] - value for key, value in paper.items()},
        "verdict": judge(mean, paper),
        "sweep": {
            str(thr): {key: value for key, value in mean_scores(rows[thr]).items() if key in ("aj", "d_avg", "oa")}
            for thr in args.vis_thr_sweep
        },
        "seconds": time.perf_counter() - start_all,
        "peak_vram_gb": max(measured_peaks) if measured_peaks else None,
        "args": vars(args),
        "dtype": str(getattr(tracker, "dtype", None)),
        "attention": getattr(tracker, "attention", None),
        "checkpoint": runtime.checkpoint_info(checkpoint),
        **runtime.run_metadata(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({key: summary[key] for key in ("dataset", "n_videos", "mean", "paper", "delta", "verdict")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
