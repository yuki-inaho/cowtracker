"""``cow-track-sparse``: tracks of given query pixels through short image clips (one forward pass per clip).

Reads a ``cow_sparse_jobs_v1`` json (see ``sparse_tracking``), validates every job before loading the model once,
and writes ``<out>/<id>.npz`` (uv [N, T, 2] float32 (x, y) in source pixels, visconf [N, T] float32, frames) per
job and ``<out>/manifest.json`` (checkpoint hash, git commit, torch version, time, peak VRAM, license note), whose
last key ``status: "complete"`` is written after every npz. CUDA only; no video is rendered.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cowtracker.toolkit import runtime
from cowtracker.toolkit.runner import DTYPES, build_runner, validate_size
from cowtracker.toolkit.sparse_tracking import track_queries, validate_jobs
from cowtracker.toolkit.video_io import read_images, resize_frames

MANIFEST_SCHEMA = "cow_sparse_tracks_v1"
LICENSE_NOTE = (
    "derived from CoWTracker (FAIR Noncommercial Research License; weights CC-BY-NC-4.0): noncommercial research only"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cow-track-sparse", description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", required=True, help="cow_sparse_jobs_v1 json")
    parser.add_argument("--out", required=True, help="output directory (must not exist)")
    parser.add_argument("--checkpoint", required=True, help="cowtracker_model.pth (never downloaded)")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        required=True,
        metavar=("H", "W"),
        help="inference size; H and W must be multiples of 112 (336 448 for 4:3)",
    )
    parser.add_argument(
        "--vram-limit-gb",
        type=float,
        default=None,
        help="per-process VRAM cap; refuses to start when less than this is free",
    )
    parser.add_argument("--device", default="cuda", help="a CUDA device (CPU inference is not an option)")
    parser.add_argument(
        "--dtype",
        choices=["auto", *DTYPES],
        default="auto",
        help="auto: fp16 on CUDA (upstream demo); bf16 needs an Ampere or newer GPU",
    )
    parser.add_argument(
        "--window-len",
        type=int,
        default=100,
        help="most frames per job (every job is one full forward pass)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if torch.device(args.device).type != "cuda":
        raise ValueError(f"--device {args.device}: cow-track-sparse runs on CUDA; CPU inference is not an option")
    size_hw = validate_size(tuple(args.size))
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"--out {out} exists; pass a new directory (partial outputs are never reused)")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"--checkpoint {args.checkpoint} is not a file (weights are never downloaded)")
    jobs_bytes = Path(args.jobs).read_bytes()  # hashed as read: the manifest names the jobs that were run
    jobs = validate_jobs(json.loads(jobs_bytes), args.window_len)

    tracker, checkpoint = build_runner(
        args.checkpoint, args.device, args.vram_limit_gb, args.window_len, "full", args.dtype
    )
    out.mkdir(parents=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    records, start_all = [], time.perf_counter()
    for index, job in enumerate(jobs):
        start = time.perf_counter()
        video = resize_frames(read_images(job.paths), size_hw)
        uv, visconf = track_queries(tracker, video, job.queries, job.source_hw)
        path = out / f"{job.id}.npz"
        np.savez(path, uv=uv, visconf=visconf, frames=np.array(job.frames))
        records.append(
            {
                "id": job.id,
                "file": path.name,
                "sha256": runtime.file_sha256(path),
                "frames": len(job.frames),
                "queries": len(job.queries),
                "seconds": time.perf_counter() - start,
            }
        )
        print(f"[{index + 1}/{len(jobs)}] {job.id}: {len(job.queries)} queries x {len(job.frames)} frames", flush=True)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "jobs_sha256": hashlib.sha256(jobs_bytes).hexdigest(),
        "size_hw": list(size_hw),
        "window_len": args.window_len,
        "dtype": str(getattr(tracker, "dtype", None)),
        "attention": getattr(tracker, "attention", None),
        "seconds": time.perf_counter() - start_all,
        "peak_vram_gb": runtime.peak_vram_gb(),
        "args": vars(args),
        "checkpoint": runtime.checkpoint_info(checkpoint),
        **runtime.run_metadata(),
        "license_note": LICENSE_NOTE,
        "jobs": records,
        "status": "complete",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out}: {len(jobs)} jobs at {size_hw[0]}x{size_hw[1]} in {manifest['seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
