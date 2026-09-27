"""``cow-track-sparse``: jobs json in, per-job query tracks (npz) and a manifest out, one runner for all jobs."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fakes import TranslatingTracker
from PIL import Image

from cowtracker.toolkit import sparse_cli
from cowtracker.toolkit.sparse_tracking import JOBS_SCHEMA

SOURCE_HW = (60, 80)


def _jobs(tmp_path, jobs=(("clip_a", 3), ("clip_b", 4))):
    """``jobs.json`` with jobs (id, frame count) over one image directory, and a stand-in checkpoint file."""
    directory = tmp_path / "rgb"
    directory.mkdir()
    for index in range(5):
        Image.fromarray(np.full((*SOURCE_HW, 3), 40 * index, np.uint8)).save(directory / f"frame_{index:06d}.png")
    spec = {
        "schema": JOBS_SCHEMA,
        "jobs": [
            {
                "id": job_id,
                "image_dir": str(directory),
                "frames": [f"frame_{index:06d}.png" for index in range(count)],
                "queries": [[0.0, 0.0], [79.0, 59.0], [20.5, 30.25]],
            }
            for job_id, count in jobs
        ],
    }
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(spec))
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    return path, checkpoint


@pytest.fixture
def built(monkeypatch):
    """Replace the checkpoint-backed runner by a translating fake and record every build."""
    calls = []

    def fake_build_runner(checkpoint, device, vram_limit_gb, window_len, mode, dtype):
        calls.append(
            {"checkpoint": checkpoint, "device": device, "window_len": window_len, "mode": mode, "dtype": dtype}
        )
        return TranslatingTracker(step=(1.0, 2.0)), checkpoint

    monkeypatch.setattr(sparse_cli, "build_runner", fake_build_runner)
    return calls


def _argv(jobs, checkpoint, out, *extra):
    return ["--jobs", str(jobs), "--checkpoint", str(checkpoint), "--out", str(out), "--size", "112", "224", *extra]


def test_writes_npz_and_manifest(tmp_path, built):
    jobs, checkpoint = _jobs(tmp_path)
    out = tmp_path / "cow"

    assert sparse_cli.main(_argv(jobs, checkpoint, out, "--window-len", "4")) == 0

    assert built == [
        {"checkpoint": str(checkpoint), "device": "cuda", "window_len": 4, "mode": "full", "dtype": "auto"}
    ]
    with np.load(out / "clip_b.npz", allow_pickle=False) as data:
        assert data["uv"].shape == (3, 4, 2) and data["uv"].dtype == np.float32
        assert data["visconf"].shape == (3, 4) and data["visconf"].dtype == np.float32
        assert data["frames"].tolist() == [f"frame_{index:06d}.png" for index in range(4)]
        step = np.array([1.0 * 80 / 224, 2.0 * 60 / 112])  # one inference-px step per frame, in source px
        np.testing.assert_allclose(data["uv"][2], [20.5, 30.25] + np.arange(4)[:, None] * step, atol=1e-4)
        np.testing.assert_allclose(data["visconf"], 0.5)
    manifest = json.loads((out / "manifest.json").read_text())
    assert list(manifest)[-1] == "status" and manifest["status"] == "complete"
    assert manifest["schema"] == sparse_cli.MANIFEST_SCHEMA
    assert manifest["license_note"].startswith("derived from CoWTracker (FAIR Noncommercial Research License")
    assert manifest["checkpoint"]["name"] == "model.pth"
    assert manifest["size_hw"] == [112, 224] and manifest["window_len"] == 4
    assert [(job["id"], job["file"], job["frames"], job["queries"]) for job in manifest["jobs"]] == [
        ("clip_a", "clip_a.npz", 3, 3),
        ("clip_b", "clip_b.npz", 4, 3),
    ]
    assert {"seconds", "sha256"} <= set(manifest["jobs"][0])
    assert {"git_commit", "torch", "attention", "dtype", "peak_vram_gb", "jobs_sha256"} <= set(manifest)
    assert manifest["jobs_sha256"] == hashlib.sha256(jobs.read_bytes()).hexdigest()


def test_refuses_existing_out(tmp_path, built):
    jobs, checkpoint = _jobs(tmp_path)
    (tmp_path / "cow").mkdir()

    with pytest.raises(FileExistsError):
        sparse_cli.main(_argv(jobs, checkpoint, tmp_path / "cow"))
    assert built == []


def test_refuses_cpu_device(tmp_path, built):
    jobs, checkpoint = _jobs(tmp_path)

    with pytest.raises(ValueError, match="CPU inference is not an option"):
        sparse_cli.main(_argv(jobs, checkpoint, tmp_path / "cow", "--device", "cpu"))
    assert built == []


def test_refuses_a_missing_checkpoint_instead_of_downloading(tmp_path, built):
    jobs, _ = _jobs(tmp_path)

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        sparse_cli.main(_argv(jobs, tmp_path / "absent.pth", tmp_path / "cow"))
    assert built == []


def test_validates_every_job_before_inference(tmp_path, built):
    jobs, checkpoint = _jobs(tmp_path, jobs=(("clip_a", 3), ("clip_b", 5)))

    with pytest.raises(ValueError, match="window"):
        sparse_cli.main(_argv(jobs, checkpoint, tmp_path / "cow", "--window-len", "4"))
    assert built == []
    assert not (tmp_path / "cow").exists()


@pytest.mark.gpu
def test_real_checkpoint_one_job(tmp_path):
    checkpoint, smoke = os.environ.get("COWTRACKER_CHECKPOINT"), os.environ.get("TA_SMOKE_ROOT")
    if not checkpoint or not Path(checkpoint).is_file() or not smoke or not Path(smoke).is_dir():
        pytest.fail("set COWTRACKER_CHECKPOINT (cowtracker_model.pth) and TA_SMOKE_ROOT (an RGB image directory)")
    from cowtracker.toolkit.video_io import image_paths

    paths = image_paths(Path(smoke))[:8]
    with Image.open(paths[0]) as image:
        width, height = image.size
    queries = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.3], [0.3, 0.8], [0.9, 0.9]]) * [width - 1, height - 1]
    queries += [0.3, -0.2]  # sub-pixel
    spec = {
        "schema": JOBS_SCHEMA,
        "jobs": [{"id": "smoke", "image_dir": smoke, "frames": [p.name for p in paths], "queries": queries.tolist()}],
    }
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(spec))
    out = tmp_path / "cow"

    sparse_cli.main(
        ["--jobs", str(jobs), "--checkpoint", checkpoint, "--out", str(out), "--size", "336", "448"]
        + ["--vram-limit-gb", "8"]
    )

    with np.load(out / "smoke.npz", allow_pickle=False) as data:
        uv, visconf = data["uv"], data["visconf"]
    assert uv.shape == (5, 8, 2) and np.isfinite(uv).all()
    assert ((visconf >= 0) & (visconf <= 1)).all()
    np.testing.assert_array_less(np.linalg.norm(uv[:, 0] - queries, axis=1), 0.5)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    print(f"seconds {manifest['jobs'][0]['seconds']:.2f}, peak VRAM {manifest['peak_vram_gb']:.2f} GB")
