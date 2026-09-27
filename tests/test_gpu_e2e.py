"""End-to-end checks with the released checkpoint on CUDA (``pytest -m gpu``; needs COWTRACKER_CHECKPOINT)."""

import os
from pathlib import Path

import mediapy
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

VIDEO = Path(__file__).resolve().parents[1] / "videos" / "bmx-bumps.mp4"
SIZE = (336, 560)
VRAM_LIMIT_GB = 30.0
GB = 1024**3
WINDOWED_250_PEAK_GB = 10.6  # measured 11.62 GB keeping the previous window, 9.69 GB without


@pytest.fixture(scope="module")
def runner():
    checkpoint = os.environ.get("COWTRACKER_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).is_file():
        pytest.fail("set COWTRACKER_CHECKPOINT to the released cowtracker_model.pth to run the gpu tests")
    from cowtracker.toolkit.runner import build_runner

    return build_runner(checkpoint, "cuda", VRAM_LIMIT_GB, window_len=100, mode="auto")[0]


def bmx(frames: int) -> torch.Tensor:
    video = np.asarray(mediapy.resize_video(mediapy.read_video(str(VIDEO))[:frames], SIZE))
    return torch.from_numpy(video).permute(0, 3, 1, 2).float().cuda()


@torch.no_grad()
def staged_and_forward_peak(model, video):
    """Predictions of the stages called by hand, of ``model(video)``, and the forward's peak VRAM per frame."""
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        images = (video / 255.0).unsqueeze(0)
        tokens, patch_idx = model.aggregator(images)
        staged = model.tracking_head(model.feature_extractor(tokens, images, patch_idx), image_size=SIZE)
        del tokens
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        predictions = model(video)
    return staged, predictions, (torch.cuda.max_memory_allocated() - before) / GB / len(video)


def test_full_forward_releases_backbone_tokens_before_head(runner):
    staged, predictions, peak_per_frame = staged_and_forward_peak(runner.model.model, bmx(32))

    assert peak_per_frame <= 0.30  # measured: 0.38 GB/frame with tokens alive, 0.256 GB/frame without
    for key in ("track", "vis", "conf"):
        assert torch.equal(predictions[key], staged[key])


def test_full_forward_keeps_only_the_dpt_layers(runner):
    staged, predictions, peak_per_frame = staged_and_forward_peak(runner.model.model, bmx(32))

    assert peak_per_frame <= 0.23  # the aggregator's own peak (0.207 GB/frame) bounds the forward
    for key in ("track", "vis", "conf"):
        assert torch.equal(predictions[key], staged[key])


def test_real_checkpoint_tracks_identity_on_frame0(runner):
    video = np.asarray(mediapy.resize_video(mediapy.read_video(str(VIDEO))[:8], SIZE))
    dense = runner(video)

    height, width = SIZE
    grid = np.stack(np.meshgrid(np.arange(width), np.arange(height), indexing="xy"), axis=-1)
    assert dense.track.shape == (8, height, width, 2)
    np.testing.assert_allclose(dense.track[0], grid, atol=1e-3)
    assert runner.calls[-1] == {"frames": 8, "mode": "full"}
    assert 0.0 <= float(dense.visconf.min()) and float(dense.visconf.max()) <= 1.0


@torch.no_grad()
def test_windowed_forward_does_not_keep_the_previous_window(runner):
    source = mediapy.read_video(str(VIDEO))
    source = np.concatenate([source, source[::-1], source])[:250]  # windows [0,100) [100,200) [200,250)
    video = torch.from_numpy(np.asarray(mediapy.resize_video(source, (224, 336)))).permute(0, 3, 1, 2).float().cuda()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        runner.model(video)
    peak_gb = (torch.cuda.max_memory_allocated() - before) / GB

    assert peak_gb <= WINDOWED_250_PEAK_GB
